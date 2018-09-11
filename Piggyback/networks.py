import torch.nn as nn
import torch
from Piggyback.custom_layers import MaskedConv2d, QuantizedConv2d
import torch.utils.model_zoo as model_zoo
import math


def create_translator():
    translator = {"bn1." + str(i) + ".": "bn1." for i in range(3)}
    
    for j in range(1, 5):
        for k in range(0, 2):
            for y in range(1, 3):
                for i in range(3):
                    key = "layer" + str(j) + "." + str(k) + ".bn" + str(y) + "."
                    translator[key + str(i) + "."] = key
    
    for j in range(2, 5):
        for i in range(3):
            key = "layer" + str(j) + ".0."
            translator[key + "bn3." + str(i) + "."] = key + "downsample.1."

    return translator


class BasicMaskedBlock(nn.Module):  # Define a residual block
    
    def __init__(self, inplanes, planes, stride=1, classes=1, first=False, quantized=False):
        super(BasicMaskedBlock, self).__init__()
        
        self.classes = classes
        self.index = 0
        
        if quantized:
            convBlock = QuantizedConv2d
        else:
            convBlock = MaskedConv2d
        
        self.conv1 = convBlock(inplanes, planes, kernel_size=3, stride=stride, padding=1, mask=classes)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(classes)])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convBlock(planes, planes, kernel_size=3, padding=1, mask=classes)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(classes)])

        self.first = inplanes != planes
        if inplanes != planes:
            self.downsample = nn.Sequential(
                  convBlock(inplanes, planes, kernel_size=1, stride=stride, mask=classes)
            )

            self.bn3 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(classes)])
        
        self.stride = stride
        
    def set_index(self, index):
        if 0 <= index < self.classes:
            self.index = index
            for i in range(self.classes):
                if index == i:
                    self.bn1[i].weight.requires_grad = True
                    self.bn1[i].bias.requires_grad = True
                    self.bn2[i].weight.requires_grad = True
                    self.bn2[i].bias.requires_grad = True
                    if self.first:
                        self.bn3[i].weight.requires_grad = True
                        self.bn3[i].bias.requires_grad = True
                else:
                    self.bn1[i].weight.requires_grad = False
                    self.bn1[i].bias.requires_grad = False
                    self.bn2[i].weight.requires_grad = False
                    self.bn2[i].bias.requires_grad = False
                    if self.first:
                        self.bn3[i].weight.requires_grad = False
                        self.bn3[i].bias.requires_grad = False
    
    def forward(self, x):
        if self.first:
            residual = self.downsample(x)
            residual = self.bn3[self.index](residual)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1[self.index](out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2[self.index](out)

        out += residual
        out = self.relu(out)

        return out


class MaskedNet(nn.Module):
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
        
        self.conv1 = convBlock(3, 64, kernel_size=7, stride=2, padding=3, mask=self.models)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(64) for i in range(self.models)])
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
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_index(self, index):
        if 0 <= index < self.models:
            self.index = index
            for m in self.modules():
                if isinstance(m, MaskedConv2d) or isinstance(m, QuantizedConv2d) or isinstance(m, BasicMaskedBlock):
                    m.set_index(index)

            for i in range(self.models):
                if i == index:
                    self.bn1[i].weight.requires_grad = True
                    self.bn1[i].bias.requires_grad = True
                else:
                    self.bn1[i].weight.requires_grad = False
                    self.bn1[i].bias.requires_grad = False
    
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
            layers.append(BasicMaskedBlock(self.in_channel, planes, stride=strides[i], classes=self.models,
                                           first=(i == 0), quantized=self.quantized))
            self.in_channel = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1[self.index](x)
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


def piggyback_net18(model_classes, pre_imagenet=True, pretrained=None, fc=True):
    model = MaskedNet(classes=model_classes, layers=[2, 2, 2, 2], fc=fc)
    if pre_imagenet:
        pre_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        model.load_state_dict(pre_dict, False)
        dic = model.state_dict()
        
        translator = create_translator()
        for na in dic:
            if na[:-6] in translator:
                name = na[:-6]
                dic[name + "weight"].copy_(pre_dict[translator[name] + "weight"])
                dic[name + "bias"] = pre_dict[translator[name] + "bias"]

        model.load_state_dict(dic)
        
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])

    print("Model pretrained loaded")
    return model


def quantized_net18(model_classes, pre_imagenet=True, pretrained=None, fc=True):
    model = MaskedNet(classes=model_classes, layers=[2, 2, 2, 2], fc=fc, quantized=True)
    if pre_imagenet:
        pre_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        model.load_state_dict(pre_dict, False)
        dic = model.state_dict()
    
        translator = create_translator()
        for na in dic:
            if na[:-6] in translator:
                name = na[:-6]
                dic[name + "weight"].copy_(pre_dict[translator[name] + "weight"])
                dic[name + "bias"] = pre_dict[translator[name] + "bias"]
    
        model.load_state_dict(dic)
        
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    
    print("Model pretrained loaded")
    return model