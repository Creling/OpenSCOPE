import torch
import torch.nn as nn
import re

layer_noise_map_db_fu5 = {
    'conv1': 0.79,                
    'layer1.0.conv1': 0.73,       
    'layer1.0.conv2': 1.02,     
    'layer1.1.conv1': 0.68,    
    'layer1.1.conv2': 1.17,    
    
    'layer2.0.conv1': 0.61,    
    'layer2.0.conv2': 0.83,     
    'layer2.1.conv1': 0.71,      
    'layer2.1.conv2': 0.86,     
    
    'layer3.0.conv1': 0.62,      
    'layer3.0.conv2': 0.72,    
    'layer3.1.conv1': 0.69,      
    'layer3.1.conv2': 0.78,     
    
    'layer4.0.conv1': 0.63,      
    'layer4.0.conv2': 1.19,      
    'layer4.1.conv1': 0.70,   
    'layer4.1.conv2': 3.49,    
    
    'default': 0.8               
}
class NoisyGeLU(nn.Module):
    """
    """
    def __init__(self, noise_std=0.8, target_snr_db=None):
        super(NoisyGeLU, self).__init__()
        self.gelu = nn.GELU()
        self.noise_std = noise_std
        self.target_snr_db = target_snr_db
    
    def forward(self, x):
        if self.target_snr_db is not None:
            signal_power = torch.mean(x**2)
            noise_power = signal_power / (10**(self.target_snr_db / 10))
            self.noise_std = torch.sqrt(noise_power).item()
        
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        x = self.gelu(x)
        x = x - self.gelu(noise)
        return x

class GeLU(nn.Module):
    """
    """
    def __init__(self):
        super(GeLU, self).__init__()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.gelu(x)

class LayerAwareNoisyGeLU(nn.Module):
    """
    """
    def __init__(self, layer_name=None, noise_std=None, noise_map=None):
        super(LayerAwareNoisyGeLU, self).__init__()
        self.gelu = nn.GELU()
        self.layer_name = layer_name
        self.noise_std = noise_std
        
        self.layer_noise_map = noise_map if noise_map is not None else layer_noise_map_db_fu5
    
    def get_layer_noise_std(self):
        print(f"Layer: {self.layer_name}, Noise Std: {self.noise_std}")
        if self.noise_std is not None:
            return self.noise_std
            
        if self.layer_name is None:
            return self.layer_noise_map.get('default', 0.8)
        
        for key in self.layer_noise_map:
            if key in self.layer_name:
                return self.layer_noise_map[key]
                
        return self.layer_noise_map.get('default', 0.8)
    
    def forward(self, x):

        signal_power = torch.mean(x**2)
        noise_power = signal_power / (10**(-5 / 10))
        noise_std = torch.sqrt(noise_power).item()
        # print(f"Layer: {self.layer_name}, Noise Std: {noise_std}")

        noise_std = torch.tensor(self.get_layer_noise_std(), device=x.device)
        print(f"Layer: {self.layer_name}, Noise Std: {noise_std}")
        
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        x = self.gelu(x)
        x = x - self.gelu(noise)
        
        return x

def replace_relu_with_noisy_gelu(module, noise_std=0.8, target_snr_db=None):
    """

    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, NoisyGeLU(noise_std=noise_std, target_snr_db=target_snr_db))
        else:
            replace_relu_with_noisy_gelu(child, noise_std, target_snr_db)

def replace_relu_with_gelu(module):
    """
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GeLU())
        else:
            replace_relu_with_gelu(child)

def replace_relu_with_layer_aware_noisy_gelu(module, parent_name="", global_noise_std=None, custom_noise_map=None):
    """
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        print(full_name)
        
        if isinstance(child, nn.ReLU):
            print(f"Replacing ReLU in {full_name} with LayerAwareNoisyGeLU")
            setattr(module, name, LayerAwareNoisyGeLU(
                layer_name=full_name, 
                noise_std=global_noise_std,
                noise_map=custom_noise_map
            ))
        else:
            replace_relu_with_layer_aware_noisy_gelu(
                child, 
                parent_name=full_name, 
                global_noise_std=global_noise_std,
                custom_noise_map=custom_noise_map
            )

def extract_model_info(filename):
    """
    """
    pattern = r"resnet18_.*noisestd_(.+)_sample_(\d+)_epoch_(\d+)\.pth"
    match = re.match(pattern, filename)
    
    if match:
        std = match.group(1)
        samples = match.group(2)
        epoch = int(match.group(3))
        return std, samples, epoch
    return None, None, None