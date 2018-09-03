import torch.nn as nn
import torch
from Piggyback.custom_layers import MaskedConv2d, QuantizedConv2d
import torch.utils.model_zoo as model_zoo


class BasicMaskedBlock(nn.Module):  # Define a residual block
    def __init__(self, inplanes, planes, stride=1, first=False, mask=1, quantized=False):
        super(BasicMaskedBlock, self).__init__()
        
        if quantized:
            convBlock = QuantizedConv2d
        else:
            convBlock = MaskedConv2d
        
        self.conv1 = convBlock(inplanes, planes, kernel_size=3, stride=stride, padding=1, mask=mask)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convBlock(planes, planes, kernel_size=3, padding=1, mask=mask)
        self.bn2 = nn.BatchNorm2d(planes)

        if first:
            self.downsample = nn.Sequential(
                  convBlock(inplanes, planes, kernel_size=1, stride=stride, mask=mask),
                  nn.BatchNorm2d(planes)
            )
        
        self.stride = stride
        self.first = first
        self.index = 0

    def set_index(self, index):
        self.index = index

    def forward(self, x):
        if self.first:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class MaskedNet(nn.Module):
    # UNICO PROBLEMA RIMASTO QUI E' CHE ABBIAMO UNA UNICA BN. PER AVERNE PIÚ DI UNA DOVREI CAMBIARE E METTERE MODULELIST
    def __init__(self, layers, classes=[1000], fc=True, quantized=False):
        super(MaskedNet, self).__init__()

        if quantized:
            convBlock = QuantizedConv2d
        else:
            convBlock = MaskedConv2d
        
        self.quantized = quantized

        self.block = convBlock
        self.in_channel = 64
        self.models = len(classes)
        kernel_size = 3
        
        self.conv1 = convBlock(3, 64, kernel_size=7, stride=2, padding=3, mask=self.models)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer_(64, layers[0])
        self.layer2 = self._make_layer_(128, layers[1], stride=2)
        self.layer3 = self._make_layer_(256, layers[2], stride=2)
        self.layer4 = self._make_layer_(512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        if fc:
            self.fc = nn.ModuleList([nn.Linear(512, c) for c in classes])
        
        self.fc_exists = fc
        
        self.index = 0
        self.set_index(self.index)

    def set_index(self, index):
        if 0 <= index < self.models:
            self.index = index
            for m in self.modules():
                if isinstance(m, MaskedConv2d) or isinstance(m, QuantizedConv2d):
                    m.set_index(index)

            if self.fc_exists:
                i = index
                for par in self.fc[i].parameters():
                    par.requires_grad = True
                # set to false others
                for i in range(0, self.models):
                    if i != index:
                        for par in self.fc[i].parameters():
                            par.requires_grad = False

    def _make_layer_(self, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for i in range(0, blocks):
            layers.append(BasicMaskedBlock(self.in_channel, planes, stride=strides[i], first=(i == 0),
                                           mask=self.models, quantized=self.quantized))
            self.in_channel = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc_exists:
            x = self.fc[self.index](x)
        return x


def piggyback_net18(model_classes, pre_imagenet=True, pretrained=None, bn=False, fc=True):
    model = MaskedNet(classes=model_classes, layers=[2, 2, 2, 2], fc=fc)
    if pre_imagenet:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'), False)
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    
    if not bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        
    print("Model pretrained loaded")
    return model


def quantized_net18(model_classes, pre_imagenet=True, pretrained=None, bn=False, fc=True):
    model = MaskedNet(classes=model_classes, layers=[2, 2, 2, 2], fc=fc, quantized=True)
    if pre_imagenet:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'), False)
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
        
    if not bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    
    print("Model pretrained loaded")
    return model