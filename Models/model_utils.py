import torch
import torch.nn as nn
from torchvision import models as torchmodels
from collections import OrderedDict

def extract_middle_layers(model, module_name):
    if module_name == 'all':
        print('returning complete model')
        return model
    else:
        list_modules = [name for name, _ in list(model.named_children())]
        modules = list(model.named_children())
        module_index = next((i for i, (name, _) in enumerate(modules) if name == module_name), None)
        if module_index is not None:
            modules = modules[:module_index+1]
        else:
            raise ValueError(f"Module {module_name} not found in model, select from {list_modules}")
        return torch.nn.Sequential(OrderedDict(modules))
    
    

class ResNetPart1(nn.Module):
    def __init__(self, original_model):
        super(ResNetPart1, self).__init__()
        self.conv1 = original_model.conv1
        

    def forward(self, x):
        x = self.conv1(x)



        return x
    
class ResNetPart2(nn.Module):
    def __init__(self, original_model):
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
    
    def forward(self, x):
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