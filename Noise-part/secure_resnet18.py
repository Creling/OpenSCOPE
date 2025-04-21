import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class FeatureExtractor:
    def __init__(self, model, layer_mapping):
        self.model = model
        self.layer_mapping = layer_mapping
        self.features = {}
        self.hooks = []

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_mapping:
                hook = module.register_forward_hook(self.save_output(name))
                self.hooks.append(hook)

    def save_output(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

detailed_noise_config = {
    'default': 0,               # Default noise

    'layer1.block0.relu1': 0.4,       
    'layer1.block0.relu2': 0.7,     
    'layer1.block1.relu1': 0.8,      
    'layer1.block1.relu2': 0.4,

    'layer2.block0.relu1': 0.5,
    # 'layer2.block0.relu2': 0,   
    'layer2.block1.relu1': 0.4,
    'layer2.block1.relu2': 0.2,

    'layer3.block0.relu1': 0.6,
    'layer3.block0.relu2': 1.1,     
    'layer3.block1.relu1': 0.6,
    'layer3.block1.relu2': 0.5,

    'layer4.block0.relu1': 0.7,
    # 'layer4.block0.relu2': 0.7,
    'layer4.block1.relu1': 2.9,
    'layer4.block1.relu2': 4.0, 
}



class SecureGELULayer(nn.Module):
    """Simulates server-client secure computation GELU layer with specific noise configuration"""
    def __init__(self, layer_id=None, noise_config=None):
        super(SecureGELULayer, self).__init__()
        self.layer_id = layer_id
        self.noise_config = noise_config if noise_config is not None else {'default': 0.1}
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # Get noise scale for this layer
        noise_scale = self.noise_config.get(self.layer_id, self.noise_config['default'])
        # print(f"Layer ID: {self.layer_id}, Noise Scale: {noise_scale}")
        
        # Server: store original input for later denoising
        original_x = x.clone()
        
        # Server: add specific noise
        noise = torch.randn_like(x) * noise_scale * 5
        noisy_x = x + noise
        
        # Client: compute GELU on noisy data
        client_gelu_output = self.gelu(noisy_x)
        
        denoised_output = client_gelu_output - self.gelu(noise)  # Denoising
        
        return denoised_output

class SecureBasicBlock(nn.Module):
    """ResNet basic residual block with secure GELU layers"""
    expansion = 1

    def __init__(self, block, noise_config, block_id):
        super(SecureBasicBlock, self).__init__()
        # Keep original convolution and batch normalization layers
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride
        
        # Create independent secure GELU layers for the two activation functions in the block
        self.secure_gelu1 = SecureGELULayer(f"{block_id}.relu1", noise_config)
        self.secure_gelu2 = SecureGELULayer(f"{block_id}.relu2", noise_config)
        
    def forward(self, x):
        identity = x
        
        # First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.secure_gelu1(out)  # First activation function with its own noise parameter
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.secure_gelu2(out)  # Second activation function with its own noise parameter
        
        return out

class SecureResNet18(nn.Module):
    def __init__(self, num_classes=1000, noise_config=None):
        super(SecureResNet18, self).__init__()
        # Set noise configuration
        self.noise_config = noise_config if noise_config is not None else {
            'default': 0.1,
            'conv1': 0.15,
            # Different noise levels for activation functions in each block
            'layer1.block0.relu1': 0.12,  # First activation function in the first block of layer 1
            'layer1.block0.relu2': 0.11,  # Second activation function in the first block of layer 1
            'layer1.block1.relu1': 0.10,
            'layer1.block1.relu2': 0.09,
            # Other layers and blocks...
        }
        
        # Load pretrained ResNet18 model
        base_model = models.resnet18(pretrained=True)
        
        # Keep the first few layers of the ResNet architecture
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.secure_gelu1 = SecureGELULayer('conv1', self.noise_config)
        self.maxpool = base_model.maxpool
        
        # Create secure blocks for each block in the residual layers
        self.layer1 = self._create_secure_layer(base_model.layer1, 'layer1')
        self.layer2 = self._create_secure_layer(base_model.layer2, 'layer2')
        self.layer3 = self._create_secure_layer(base_model.layer3, 'layer3')
        self.layer4 = self._create_secure_layer(base_model.layer4, 'layer4')
        
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc
        
        if num_classes != 1000:
            self.fc = nn.Linear(self.fc.in_features, num_classes)
    
    def _create_secure_layer(self, layer, layer_prefix):
        """Convert original residual layer to secure residual layer, handling multiple activation functions in each block"""
        secure_blocks = nn.Sequential()
        for i, block in enumerate(layer):
            block_id = f"{layer_prefix}.block{i}"
            secure_block = SecureBasicBlock(block, self.noise_config, block_id)
            secure_blocks.add_module(str(i), secure_block)
        return secure_blocks
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.secure_gelu1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

