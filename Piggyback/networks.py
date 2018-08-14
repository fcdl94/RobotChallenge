import torch.nn as nn
import torch
from Piggyback.custom_layers import MaskedConv2d


# Define a residual block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if first:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))
        self.stride = stride
        self.first = first

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

        out = out + residual
        out = self.relu(out)

        return out


# Define a residual block
class BasicMaskedBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, first=False, model_size=1):
        super(BasicMaskedBlock, self).__init__()
        self.conv1 = MaskedConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1,
                                  bias=False, model_size=model_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=kernel_size, padding=1, bias=False, model_size=model_size)
        self.bn2 = nn.BatchNorm2d(planes)

        if first:
            self.downsample = nn.Sequential(MaskedConv2d(inplanes, planes, kernel_size=1, stride=stride,
                                                         bias=False, model_size=model_size),
                                            nn.BatchNorm2d(planes))
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


class WideResNet(nn.Module):
    def __init__(self, resnet_block, widening_factor=4, kernel_size=3, classes=[1000]):
        super(WideResNet, self).__init__()
        
        self.block = nn.Conv2d
        self.in_channel = 16
        
        self.conv1 = self.block(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer_(resnet_block, 64, kernel_size, widening_factor, stride=2)
        self.layer2 = self._make_layer_(resnet_block, 128, kernel_size, widening_factor, stride=2)
        self.layer3 = self._make_layer_(resnet_block, 256, kernel_size, widening_factor, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.ModuleList([nn.Linear(256, c) for c in classes])
        self.index = 0
        
    def set_index(self, index):
        if index < len(self.fc):
            self.index = index
            i = index
            for par in self.fc[i].parameters():
                par.requires_grad = True

            for i in range(0, self.models):
                if i != index:
                    for par in self.fc[i].parameters():
                        par.requires_grad = False
                        
    def add_task(self, module):
        self.fc.append(module)
        return len(self.fc) - 1  # return actual index of the added module
        
    def _make_layer_(self, resnet_block, planes, kernel_size, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for i in range(0, blocks):
            layers.append(resnet_block(self.in_channel, planes, kernel_size, stride=strides[i], first=(i == 0)))
            self.in_channel = planes

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc[self.index](x)
        return x


class PiggybackNet(nn.Module):
    def __init__(self, widening_factor=4, kernel_size=3, classes=[1000]):
        super(PiggybackNet, self).__init__()

        self.block = MaskedConv2d
        self.resnet_block = BasicMaskedBlock
        self.in_channel = 16
        self.models = len(classes)

        self.conv1 = MaskedConv2d(3, self.in_channel, kernel_size=kernel_size, stride=1,
                                  padding=1, model_size=self.models)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU()
        self.layer1 = self._make_layer_(self.resnet_block, 64, kernel_size, widening_factor, stride=2)
        self.layer2 = self._make_layer_(self.resnet_block, 128, kernel_size, widening_factor, stride=2)
        self.layer3 = self._make_layer_(self.resnet_block, 256, kernel_size, widening_factor, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.ModuleList([nn.Linear(256, c) for c in classes])
        self.index = 0

    def set_index(self, index):
        if index < len(self.fc):
            self.index = index
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_index(index)
            # self.conv1.set_index(index)
            # for mod in self.layer1.modules():
            #     if isinstance(mod, MaskedConv2d):
            #         mod.set_index(index)
            # for mod in self.layer2.modules():
            #     if isinstance(mod, MaskedConv2d):
            #         mod.set_index(index)
            # for mod in self.layer3.modules():
            #     if isinstance(mod, MaskedConv2d):
            #         mod.set_index(index)
            # set to true correct modules
            i = index
            for par in self.fc[i].parameters():
                par.requires_grad = True
            # set to false others
            for i in range(0, self.models):
                if i != index:
                    for par in self.fc[i].parameters():
                        par.requires_grad = False

    def add_task(self, module):
        self.fc.append(module)
        return len(self.fc) - 1  # return actual index of the added module

    def _make_layer_(self, block, planes, kernel_size, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.in_channel, planes, kernel_size,
                                stride=strides[i], first=(i == 0), model_size=self.models))
            self.in_channel = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc[self.index](x)
        return x


def wide_resnet(model_classes, pretrained=None, frozen=False, imagenet_old=False):
    model = WideResNet(BasicBlock, classes=model_classes)
    if pretrained:
        old_state = torch.load(pretrained)['state_dict']
        state = model.state_dict()
        state.update(old_state)
        model.load_state_dict(state, False)

        if imagenet_old:
            dict_fc = dict(model.fc.named_parameters())
            dict_fc["0.weight"].data.copy_(old_state["fc.weight"].data)
            dict_fc["0.bias"].data.copy_(old_state["fc.bias"].data)

        print("Model pretrained loaded")

    if frozen:
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model


def piggyback_net(model_classes, pretrained=None, bn=False):
    model = PiggybackNet(classes=model_classes)
    if pretrained:
        old_state = torch.load(pretrained)['state_dict']
        state = model.state_dict()
        state.update(old_state)
        model.load_state_dict(state, False)

        # if imagenet_old:
        #     dict_fc = dict(model.fc.named_parameters())
        #     dict_fc["0.weight"].data.copy_(old_state["fc.weight"].data)
        #     dict_fc["0.bias"].data.copy_(old_state["fc.bias"].data)

        if not bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        
        print("Model pretrained loaded")
    return model

