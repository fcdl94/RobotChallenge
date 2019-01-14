import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        return x


def resnet18(pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), False)
    return model


class RGBDResnet(nn.Module):
    def __init__(self, rgb_network, depth_network, feautres, classes):
        super(RGBDResnet, self).__init__()
        self.net1 = rgb_network
        self.net2 = depth_network
        self.fc = nn.Linear(feautres*2, classes)
        
        self.classes = classes
        
    def forward(self, x):
        
        rgb, depth = x
        rgb = self.net1(rgb)
        depth = self.net2(depth)
        
        out = torch.stack((rgb, depth), 1).view(rgb.shape[0], -1)
        
        out = self.fc(out)
        
        return out
    
    
def double_resnet18(classes, pretrained=None):
    
    rgb_net = resnet18(classes)
    
    depth_net = resnet18(classes)
    
    net = RGBDResnet(rgb_net, depth_net, 512, classes)
    
    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])
    
    return net


class RGBDCustomNet(nn.Module):
    def __init__(self, rgb_network, depth_network, features, classes):
        super(RGBDCustomNet, self).__init__()
        self.net1 = rgb_network
        self.net2 = depth_network
        self.fc = nn.ModuleList([nn.Linear(features*2, c) for c in classes])
        
        self.classes = classes
        self.index = 0
        
        # self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 0.0001
        for f in self.fc:
            f.weight.data.uniform_(-stdv, stdv)
            if f.bias is not None:
                f.bias.data.uniform_(-stdv, stdv)
        
    def set_index(self, index):
        self.net1.set_index(index)
        self.net2.set_index(index)
        self.index = index

        i = index
        for par in self.fc[i].parameters():
            par.requires_grad = True
        # set to false others
        for i in range(0, len(self.classes)):
            if i != index:
                for par in self.fc[i].parameters():
                    par.requires_grad = False
    
    def forward(self, x):
        rgb, depth = x
        rgb = self.net1(rgb)
        depth = self.net2(depth)
        
        out = torch.stack((rgb, depth), 1).view(rgb.shape[0], -1)
        
        out = self.fc[self.index](out)
        
        return out


def double_piggyback18(classes, index, pretrained=None):
    from Piggyback.networks import piggyback_net18

    rgb_net = piggyback_net18(classes, fc=False)

    depth_net = piggyback_net18(classes, fc=False)
    
    net = RGBDCustomNet(rgb_net, depth_net, 512, classes)
    
    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])
    
    net.set_index(index)
    
    return net


def double_quantized18(classes, index, pretrained=None):
    from Piggyback.networks import quantized_net18
    
    rgb_net = quantized_net18(classes, fc=False)
    
    depth_net = quantized_net18(classes, fc=False)
    
    net = RGBDCustomNet(rgb_net, depth_net, 512, classes)
    
    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])
    
    net.set_index(index)
    
    return net


def double_serial18(classes, index, pretrained=None):
    from Rebuffi.networks import rebuffi_net18
    
    rgb_net = rebuffi_net18(classes, serie=True, pre_imagenet=True, fc=False)
    
    depth_net = rebuffi_net18(classes, serie=True, pre_imagenet=True,  fc=False)
    
    net = RGBDCustomNet(rgb_net, depth_net, 512, classes)
    
    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])
    
    net.set_index(index)
    
    return net


def double_parallel18(classes, index, pretrained=None):
    from Rebuffi.networks import rebuffi_net18
    
    rgb_net = rebuffi_net18(classes, serie=False, pre_imagenet=True, fc=False)
    
    depth_net = rebuffi_net18(classes, serie=False, pre_imagenet=True, fc=False)
    
    net = RGBDCustomNet(rgb_net, depth_net, 512, classes)
    
    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])
    
    net.set_index(index)
    
    return net


def double_combined18(classes, index, order, pretrained=None):
    from CombinedNet.networks import combined_net18

    rgb_net = combined_net18(classes, pre_imagenet=True, fc=False, order=order)

    depth_net = combined_net18(classes, pre_imagenet=True, fc=False, order=order)

    net = RGBDCustomNet(rgb_net, depth_net, 512, classes)

    if pretrained:
        net.load_state_dict(torch.load(pretrained)["state_dict"])

    net.set_index(index)

    return net
