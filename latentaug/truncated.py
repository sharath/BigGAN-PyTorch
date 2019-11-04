import torch
import torch.nn as nn
import torchvision.models as models


class TruncatedModel(nn.Module):
    def __init__(self, model, device='cpu'):
        super(TruncatedModel, self).__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, x):
        pass
        


class TruncatedResNet18(nn.Module):
    def __init__(self, device='cpu'):
        super(TruncatedResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True).to(device)
        
    def forward(self, x, n):
        if n > 0:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        if n > 1:
            x = self.model.layer1(x)
        if n > 2:
            x = self.model.layer2(x)
        if n > 3:
            x = self.model.layer3(x)
        if n > 4:
            x = self.model.layer4[0].conv1(x)
            #x = self.model.layer4[0].relu(x)
        return x