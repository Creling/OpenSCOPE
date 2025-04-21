import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from medmnist import INFO, OrganAMNIST
from tqdm import tqdm
from noisygelu import replace_relu_with_noisy_gelu


# 1. Parameter settings
data_flag = 'organamnist'
download = True
batch_size = 64
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset information
info = INFO[data_flag]
n_classes = len(info['label'])  # OrganAMNIST has 11 classes

# 3. Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # OrganAMNIST is single channel
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Simple data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# 4. Load datasets
train_dataset = OrganAMNIST(split='train', transform=transform, download=download)
val_dataset   = OrganAMNIST(split='val', transform=transform, download=download)
test_dataset  = OrganAMNIST(split='test', transform=transform, download=download)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. Build model: resnet18 + single channel input + 11 class output
model = models.resnet18(weights="IMAGENET1K_V1")
replace_relu_with_noisy_gelu(model)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

# 6. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 7. Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.view(-1).long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

        # Update progress bar status
        loop.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total

# 8. Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total

# 9. Training main loop
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# 10. Test set evaluation
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"[Test] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# 11. Save model
torch.save(model.state_dict(), "resnet18_organamnist_noisygelu_db_-1.pth")
# torch.save(model.state_dict(), "resnet18_organamnist_gelu.pth")
