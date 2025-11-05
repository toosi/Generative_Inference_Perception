"""Model utility functions."""

from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
from torchvision import models as torchmodels


def extract_middle_layers(model: nn.Module, module_name: Union[str, int]) -> nn.Module:
    """Extract middle layers from a model up to a specified module.
    
    Args:
        model: PyTorch model to extract layers from.
        module_name: Module name or 'all' for full model.
        
    Returns:
        Truncated model up to the specified module.
        
    Raises:
        ValueError: If module_name is not found in the model.
    """
    if module_name == 'all':
        return model
    else:
        list_modules = [name for name, _ in list(model.named_children())]
        modules = list(model.named_children())
        module_index = next(
            (i for i, (name, _) in enumerate(modules) if name == module_name),
            None
        )
        if module_index is not None:
            modules = modules[:module_index + 1]
        else:
            raise ValueError(
                f"Module {module_name} not found in model, "
                f"select from {list_modules}"
            )
        return nn.Sequential(OrderedDict(modules))
    
    

class ResNetPart1(nn.Module):
    """First part of ResNet (conv1 layer)."""
    
    def __init__(self, original_model: nn.Module):
        """Initialize ResNet part 1.
        
        Args:
            original_model: Original ResNet model.
        """
        super(ResNetPart1, self).__init__()
        self.conv1 = original_model.conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv1.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.conv1(x)
    
class ResNetPart2(nn.Module):
    """Second part of ResNet (remaining layers)."""
    
    def __init__(self, original_model: nn.Module):
        """Initialize ResNet part 2.
        
        Args:
            original_model: Original ResNet model.
        """
        super(ResNetPart2, self).__init__()
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through remaining layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = torchmodels.resnet18()
    print([name for name, _ in list(model.named_children())])
    modelr = extract_middle_layers(model, 'layer3')
    print(modelr)