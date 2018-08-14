import torch.nn as nn
import math
import torch
import torchvision
import torch.utils.model_zoo as model_zoo


def resnet18(fc_classes=1000, pretrained=None):
    """Constructs a ResNet-18 model.
    Args:
        fc_classes (int): The number of classes the model has to output. E.g. ImageNet12 has 1000 classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = torchvision.models.resnet18(False)
    if pretrained:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, fc_classes)
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    else:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, fc_classes)
    
    return model
