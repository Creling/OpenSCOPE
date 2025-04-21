import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
from torchvision.models import resnet50
from medmnist import OrganAMNIST
from tqdm import tqdm
from noisygelu_v2 import replace_relu_with_noisy_gelu, replace_relu_with_gelu
import wandb  # Import wandb for tracking experiments
from random_word import RandomWords
from myUtils import stratified_split_dataset_by_count, stratified_sample_dataset
from noisy_activation_utils import replace_relu_with_layer_aware_noisy_gelu, layer_noise_map_db_fu5_resnet50

r = RandomWords()
# stragety = "independent"
# stragety = "e2e"
stragety = "joint"

# Initialize wandb
wandb.init(
    project="resnet50_model-stealing-resnet_fixsamples",
    name = f"{stragety}-{r.get_random_word()}-{r.get_random_word()}",
    config={
        "architecture": "resnet50",
        "dataset": "OrganAMNIST",
        "epochs": 50,
        "learning_rate": 1e-3,
        "target_snr_db": -1,
        "target_model_path": "resnet50_organamnist_gelu_noisygelu_db_-5.pth",
        "noise_std": "layer_noise_map_db_fu5_resnet50",
        # "noise_std": None,
        "target_snr_db": None,
        "strategy": stragety,  # "e2e" for end-to-end, "joint" for joint training
        "sample_per_cls": 30
    }
)

config = wandb.config

# ----------------------------
# Load target model
# ----------------------------
target_model = resnet50(pretrained=True)
replace_relu_with_layer_aware_noisy_gelu(target_model)

# Modify the first convolutional layer to handle single-channel OrganAMNIST input
target_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the final fully connected layer to handle 11 classes for OrganAMNIST
target_model.fc = nn.Linear(target_model.fc.in_features, 11)

target_model.eval()  

print(torch.load(config.target_model_path, weights_only=True)['model_state_dict'].keys())
target_model.load_state_dict(torch.load(config.target_model_path, weights_only=True)['model_state_dict'])

# ----------------------------
# Data preparation and transformations
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # No data augmentation for validation set
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Create datasets
data_flag = 'organamnist'
download = True


testset = OrganAMNIST(root='./data', split='test', transform=transform, download=download)
labels = [testset[i][1] for i in range(len(testset))]

queryset = torch.utils.data.Subset(testset, range(len(testset) // 2))  # Use first half of samples
valset = torch.utils.data.Subset(testset, range(len(testset) // 2, len(testset)))  # Use remaining samples for validation
sub_queryset = stratified_sample_dataset(testset, config.sample_per_cls)

print(f"queryset length: {len(sub_queryset)}")
trainloader = torch.utils.data.DataLoader(sub_queryset, batch_size=64, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

# Record dataset sizes
wandb.config.update({
    "query_set_size": len(queryset),
    "val_set_size": len(valset),
    "batch_size": 64
})

# ----------------------------
# Register hooks for all convolutional and linear layers of the target model to capture inputs/outputs during forward pass
# ----------------------------
activation_storage = {}

def get_activation_hook(name):
    def hook(module, input, output):
        # Store inputs and outputs for current layer (detach and clone to prevent gradient issues and memory sharing)
        activation_storage[name] = (input[0].detach().clone(), output.detach().clone())
    return hook

hooks = []
for name, module in target_model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if "downsample" in name:
            print(f"hook pass {name}")
            continue
        if "fc" in name:
            print(f"hook pass {name}")
            continue
        hook = module.register_forward_hook(get_activation_hook(name))
        hooks.append(hook)

# ----------------------------
# Create approximator dictionary for each convolutional and linear layer
# ----------------------------
approx_layers = {}

for name, module in target_model.named_modules():
    if isinstance(module, nn.Conv2d):
        if "downsample" in name:
            print(f"pass {name}")
            continue
        # Create an approximator for each convolutional layer
        approx_layers[name] = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=(module.bias is not None)
        )

    # if isinstance(module, nn.Linear):
    #     approx_layers[name] = nn.Linear(
    #         in_features=module.in_features,
    #         out_features=module.out_features,
    #         bias=(module.bias is not None)
    #     )

# ----------------------------
# Construct the end-to-end approximation model (resnet50)
# ----------------------------
approx_model = resnet50(pretrained=False)
replace_relu_with_gelu(approx_model) 

# Modify the first convolutional layer to handle single-channel OrganAMNIST input
approx_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the final fully connected layer to handle 11 classes for OrganAMNIST
approx_model.fc = nn.Linear(approx_model.fc.in_features, 11)

def replace_conv_layers(model, approx_layers, prefix=''):
    """
    Recursively traverse the model and replace Conv2d and Linear layers with the corresponding
    approximator from approx_layers
    """
    for name, module in model.named_children():
        full_name = prefix + ('.' if prefix else '') + name
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if full_name in approx_layers:
                print(f"Replacing: {full_name}")
                setattr(model, name, approx_layers[full_name])
        else:
            replace_conv_layers(module, approx_layers, full_name)

replace_conv_layers(approx_model, approx_layers)
approx_model.train()

# Save initial model architecture to wandb
wandb.watch(approx_model)

# ----------------------------
# Setup optimizer
# ----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(approx_model.parameters(), lr=1e-3)

# Define evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ----------------------------
# Training loop: Fit both independent layer approximation and end-to-end approximation
# ----------------------------
num_epochs = config.epochs  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model.to(device)
approx_model.to(device)
for layer in approx_layers.values():
    layer.to(device)

# For tracking best model
best_acc = 0.0
best_epoch = 0

# Evaluate target model accuracy on validation set (for reference)
target_acc = evaluate_model(target_model, valloader, device)
print(f"Target model accuracy on validation set: {target_acc:.2f}%")
wandb.log({"target_model_accuracy": target_acc})

# Track loss for each layer
layer_losses = {name: 0.0 for name in approx_layers}

for epoch in range(num_epochs):
    # Training phase
    approx_model.train()
    running_loss = 0.0
    running_layer_losses = {name: 0.0 for name in approx_layers}
    running_end2end_loss = 0.0
    
    # Create progress bar with tqdm
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for i, (inputs, labels) in enumerate(pbar):
        # if config.strategy == "independent":
        #     temp = {
        #         "inputs": inputs,
        #         "labels": labels
        #     }
        #     torch.save(temp, f"datasample_index_{i}_epoch_{epoch}.pth")
        inputs = inputs.to(device)
        optimizer.zero_grad()
        activation_storage.clear()  # Clear previous records
        
        # Forward pass: Capture intermediate activations through target model and get final output
        with torch.no_grad():
            target_out = target_model(inputs)

        # Calculate approximation loss for each layer (local fitting)
        loss_layers = torch.tensor(0.0, device=device)

        if config.strategy == "joint" or config.strategy == "independent" :
            for name, approx_layer in approx_layers.items():
                if name in activation_storage:
                    in_act, target_act = activation_storage[name]
                    # Create new tensors to avoid issues with in-place operations
                    in_act_noisy = in_act.clone()
                    target_act_noisy = target_act.clone()

                    if config.noise_std != None:
                        in_act_noisy = in_act_noisy + torch.randn_like(in_act) * eval(config.noise_std)[name]
                        target_act_noisy = target_act_noisy + torch.randn_like(target_act) * eval(config.noise_std)[name]

                    approx_out = approx_layer(in_act_noisy)
                    layer_loss = criterion(approx_out, target_act_noisy)
                    loss_layers = loss_layers + layer_loss
                    running_layer_losses[name] += layer_loss.item()
        
        loss_end2end = torch.tensor(0.0, device=device)

        if config.strategy == "joint" or config.strategy == "e2e" :
            # End-to-end approximation loss: Mean squared error between approx model output and target model output
            approx_out = approx_model(inputs)
            loss_end2end = criterion(approx_out, target_out)
            running_end2end_loss += loss_end2end.item()
        
        # Total loss: Layer-level loss + end-to-end loss
        loss = loss_layers + loss_end2end
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update tqdm progress bar to show current batch loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Log batch loss every 100 batches
        if i % 100 == 99:
            batch_metrics = {
                "train/batch_loss": loss.item(),
                "train/batch_end2end_loss": loss_end2end.item(),
                "train/batch_layers_loss": loss_layers.item()
            }
            wandb.log(batch_metrics)
    
    # Calculate average epoch loss
    epoch_loss = running_loss / len(trainloader)
    epoch_end2end_loss = running_end2end_loss / len(trainloader)
    
    # Average loss for each layer
    epoch_layer_losses = {name: loss / len(trainloader) for name, loss in running_layer_losses.items()}
    
    # Validate once per epoch
    approx_acc = evaluate_model(approx_model, valloader, device)
    
    # Print training loss and validation accuracy for current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Approx Acc: {approx_acc:.2f}% - Target Acc: {target_acc:.2f}%")
    
    # Log to wandb
    epoch_metrics = {
        "epoch": epoch + 1,
        "train/loss": epoch_loss,
        "train/end2end_loss": epoch_end2end_loss,
        "val/approx_accuracy": approx_acc,
        "val/target_accuracy": target_acc,
        "val/accuracy_gap": target_acc - approx_acc
    }
    
    # Add layer loss to wandb
    for name, loss in epoch_layer_losses.items():
        layer_name = name.replace(".", "_")  # wandb-compatible name
        epoch_metrics[f"layer_loss/{layer_name}"] = loss
    
    wandb.log(epoch_metrics)
        
    # Save model checkpoint to wandb
    # wandb.save(checkpoint_path)
    
    # Save best model
    if approx_acc > best_acc:
        best_acc = approx_acc
        best_epoch = epoch + 1
        best_model_path = "resnet50_organamnist_approx_best.pth"
        torch.save(approx_model.state_dict(), best_model_path)
        # wandb.save(best_model_path)
        print(f"Saved best model, accuracy: {best_acc:.2f}%")

# Save current epoch checkpoint to wandb
checkpoint_path = f"resnet50_organamnist_{config.strategy}_noisestd_{config.noise_std}_sample_{config.sample_per_cls}_epoch_{num_epochs}.pth"
torch.save(approx_model.state_dict(), checkpoint_path)

# Remove hooks to prevent memory leaks
for hook in hooks:
    hook.remove()

# Load best model for final evaluation
approx_model.load_state_dict(torch.load("resnet50_organamnist_approx_best.pth"))

# Final evaluation on test set
final_target_acc = evaluate_model(target_model, valloader, device)
final_approx_acc = evaluate_model(approx_model, valloader, device)

# Log final results
final_metrics = {
    "final/target_accuracy": final_target_acc,
    "final/approx_accuracy": final_approx_acc,
    "final/accuracy_gap": final_target_acc - final_approx_acc,
    "best_epoch": best_epoch,
    "best_accuracy": best_acc
}
wandb.log(final_metrics)

print("\n========== Final Results ==========")
print(f'Target model accuracy: {final_target_acc:.2f}%')
print(f'Approx model accuracy: {final_approx_acc:.2f}% (Best model from epoch {best_epoch})')
print(f'Accuracy gap between target and approx model: {final_target_acc - final_approx_acc:.2f}%')
print("Model stealing training complete!")

# Finish experiment
wandb.finish()