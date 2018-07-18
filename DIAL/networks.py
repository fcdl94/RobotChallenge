import torch.nn as nn
import math
import torch
import DIAL.utils

#TODO modificarlo in modo che non abbia da fare i set index ma direttamente modificando la funzione chiamabile tale che modifichi la forward.
#   Controllare non sia una follia, ma penso di no.

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'RODS1': '/home/fabio/robot_challenge/RobotChallenge/RODx_models/S1LR23BS64t1.pth'
}


class DomainAdaptationLayer(nn.Module):
    def __init__(self, planes):
        super(DomainAdaptationLayer, self).__init__()
        self.bn_source = nn.BatchNorm2d(planes)
        self.bn_target = nn.BatchNorm2d(planes)
        self.index = 0
  
    def set_domain(self, source=True):
        self.index = 0 if source else 1
  
    def forward(self, x, index=True):
        if index:
            out = self.bn_source(x)
        else:
            out = self.bn_target(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DomainAdaptationLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DomainAdaptationLayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.index = 0

    def set_domain(self, source=True):
        self.index = 0 if source else 1
        if self.downsample is not None:
            self.downsample[1].set_domain(source)
        self.bn1.set_domain(source)
        self.bn2.set_domain(source)

    def forward(self, x, index):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, index)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, index)

        if self.downsample is not None:
            residual = self.downsample(x, index)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DomainAdaptationLayer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.index = 0
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_domain(self, source=True):
        self.index = 0 if source else 1
        self.bn1.set_domain(source)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer.modules():
                if isinstance(block, DomainAdaptationLayer) or isinstance(block, BasicBlock):
                    block.set_domain(source)
            
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                DomainAdaptationLayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x, index):
        x = self.conv1(x)
        x = self.bn1(x, index)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, index)
        x = self.layer2(x, index)
        x = self.layer3(x, index)
        x = self.layer4(x, index)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def load_pretrained(self, state_dict):
        dict_model = self.state_dict()
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


def resnet18(fc_classes=1000, pretrained=None):
    """Constructs a ResNet-18 model.
    Args:
        fc_classes (int): The number of classes the model has to output. E.g. ImageNet12 has 1000 classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=fc_classes)
    if pretrained:
        model.load_pretrained(torch.load(pretrained)['state_dict'])
        # DEBUG
        assert DIAL.utils.check_equals_bn(model.state_dict(), torch.load(pretrained)['state_dict'])
    return model
