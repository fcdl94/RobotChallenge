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


def load_pretrained(model, state_dict):
    dict_model = model.state_dict()
    for key in state_dict.keys():
        if "bn" in key:
            if "weight" in key:
                dict_model[key[:-6] + "bn_source.weight"].data.copy_(state_dict[key].data)
                dict_model[key[:-6] + "bn_target.weight"].data.copy_(state_dict[key].data)
            elif "bias" in key:
                dict_model[key[:-4] + "bn_source.bias"].data.copy_(state_dict[key].data)
                dict_model[key[:-4] + "bn_target.bias"].data.copy_(state_dict[key].data)
        elif 'downsample' in key:
            if "0.weight" in key or "0.bias" in key:
                dict_model[key].data.copy_(state_dict[key].data)
            elif "1.weight" in key:
                dict_model[key[:-6] + "bn_source.weight"].data.copy_(state_dict[key].data)
                dict_model[key[:-6] + "bn_target.weight"].data.copy_(state_dict[key].data)
            elif "1.bias" in key:
                dict_model[key[:-4] + "bn_source.bias"].data.copy_(state_dict[key].data)
                dict_model[key[:-4] + "bn_target.bias"].data.copy_(state_dict[key].data)
        else:
            dict_model[key].data.copy_(state_dict[key].data)
