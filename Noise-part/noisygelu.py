import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class NoisyGeLU(nn.Module):
    def __init__(self):
        super(NoisyGeLU, self).__init__()
        self.gelu = nn.GELU()
        self.target_snr_db = -1

    def forward(self, x):
        signal_power = torch.mean(x ** 2)
        noise_std = torch.sqrt(signal_power / (10 ** (self.target_snr_db / 10)))
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        x = self.gelu(x)
        x = x - self.gelu(noise)
        return x
    
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(x)
        return x
    
def  replace_relu_with_noisy_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, NoisyGeLU())
        else:
            replace_relu_with_noisy_gelu(child)

def replace_relu_with_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GeLU())
        else:
            replace_relu_with_gelu(child)