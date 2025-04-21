cd import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import OrganAMNIST
from medmnist import INFO
import numpy as np
from noisy_activation_utilsv2 import replace_relu_with_gelu
from secure_resnet18 import SecureResNet18, FeatureExtractor, detailed_noise_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_model(base_model, num_classes):
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    return base_model

def prepare_data(batch_size):
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = OrganAMNIST(split='train', transform=data_transform, download=True)
    val_dataset = OrganAMNIST(split='val', transform=data_transform, download=True)
    test_dataset = OrganAMNIST(split='test', transform=data_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.view(-1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_secure_resnet18():
    batch_size = 32
    num_classes = 11 
    epochs = 25
    learning_rate = 1e-3

    base_model = models.resnet18(pretrained=True)
    base_model = prepare_model(base_model, num_classes)

    replace_relu_with_gelu(base_model)  
    base_model.load_state_dict(torch.load("resnet18_organamnist_gelu.pth"))
    base_model = base_model.to(device)

    secure_resnet18 = SecureResNet18(num_classes=num_classes, noise_config=detailed_noise_config)
    secure_resnet18 = prepare_model(secure_resnet18, num_classes)
    secure_resnet18 = secure_resnet18.to(device)
    
    train_loader, val_loader, test_loader = prepare_data(batch_size)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(secure_resnet18.parameters(), lr=learning_rate)

    layer_mapping = {
        "conv1": "conv1", 
        "bn1": "bn1", 
        # Layer1
        "layer1.0.conv1": "layer1.0.conv1",
        "layer1.0.bn1": "layer1.0.bn1",
        "layer1.0.conv2": "layer1.0.conv2",
        "layer1.0.downsample": "layer1.0.downsample",
        "layer1.1.conv1": "layer1.1.conv1",
        "layer1.1.bn1": "layer1.1.bn1",
        "layer1.1.conv2": "layer1.1.conv2",
        # Layer2
        "layer2.0.conv1": "layer2.0.conv1",
        "layer2.0.bn1": "layer2.0.bn1",
        "layer2.0.conv2": "layer2.0.conv2",
        "layer2.0.downsample": "layer2.0.downsample",
        "layer2.1.conv1": "layer2.1.conv1",
        "layer2.1.bn1": "layer2.1.bn1",
        "layer2.1.conv2": "layer2.1.conv2",
        # Layer3
        "layer3.0.conv1": "layer3.0.conv1",
        "layer3.0.bn1": "layer3.0.bn1",
        "layer3.0.conv2": "layer3.0.conv2",
        "layer3.0.downsample": "layer3.0.downsample",
        "layer3.1.conv1": "layer3.1.conv1",
        "layer3.1.bn1": "layer3.1.bn1",
        "layer3.1.conv2": "layer3.1.conv2",
        # Layer4
        "layer4.0.conv1": "layer4.0.conv1",
        "layer4.0.bn1": "layer4.0.bn1",
        "layer4.0.conv2": "layer4.0.conv2",
        "layer4.0.downsample": "layer4.0.downsample",
        "layer4.1.conv1": "layer4.1.conv1",
        "layer4.1.bn1": "layer4.1.bn1",
        "layer4.1.conv2": "layer4.1.conv2",
    }

    resnet_feature_extractor = FeatureExtractor(base_model, layer_mapping)
    secure_feature_extractor = FeatureExtractor(secure_resnet18, layer_mapping)

    resnet_feature_extractor.register_hooks()
    secure_feature_extractor.register_hooks()

    best_val_accuracy = 0.0
    for epoch in range(epochs):
        secure_resnet18.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.view(-1).to(device)

            # Forward pass for both models
            with torch.no_grad():
                base_model(images)
            secure_resnet18(images) 

            feature_loss = 0
            feature_loss_1 = 0
            feature_loss_2 = 0
            for layer_name in layer_mapping:
                if "conv1" in layer_name or "bn1" in layer_name:  # conv1 -> bn1 
                    feature_loss_1 = mse_loss(
                        secure_feature_extractor.features[layer_name],
                        resnet_feature_extractor.features[layer_name]
                    )
                elif "conv2" in layer_name:  # conv2 + downsample
                    secure_combined_output = secure_feature_extractor.features[layer_name]
                    if f"{layer_name.replace('conv2', 'downsample')}" in secure_feature_extractor.features:
                        secure_combined_output_2 = secure_feature_extractor.features[f"{layer_name.replace('conv2', 'downsample')}"]
                        secure_combined_output = secure_combined_output + secure_combined_output_2

                    resnet_combined_output = resnet_feature_extractor.features[layer_name]
                    if f"{layer_name.replace('conv2', 'downsample')}" in resnet_feature_extractor.features:
                        resnet_combined_output_2 = resnet_feature_extractor.features[f"{layer_name.replace('conv2', 'downsample')}"]
                        resnet_combined_output = resnet_combined_output + resnet_combined_output_2

                    feature_loss_2 = mse_loss(secure_combined_output, resnet_combined_output)

            feature_loss = feature_loss_1 + feature_loss_2

            outputs = secure_resnet18(images)
            task_loss = ce_loss(outputs, labels)

            total_loss = feature_loss * 2 + task_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate(secure_resnet18, val_loader, ce_loss, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(secure_resnet18.state_dict(), "best_secure_resnet18v5_2.pth")
        
        torch.save(secure_resnet18.state_dict(), f"secure_resnet18_organamnist_noisygelu_v5_epoch{epoch}.pth")

    resnet_feature_extractor.clear_hooks()
    secure_feature_extractor.clear_hooks()

    test_loss, test_accuracy = evaluate(secure_resnet18, test_loader, ce_loss, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_secure_resnet18()