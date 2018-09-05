import torch
import torch.nn as nn
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


def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class ParallelAdapterModule(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, nb_tasks=1):
        super(ParallelAdapterModule, self).__init__()
        
        self.task = 0
        self.nb_task = nb_tasks
        self.conv = nn.ModuleList([conv1x1_fonc(in_planes, planes, stride) for i in range(nb_tasks)])
    
    def set_index(self, index):
        self.task = index
        for i in range(self.nb_task):
            if i == index:
                self.conv[i].weight.requires_grad = True
            else:
                self.conv[i].weight.requires_grad = False
    
    def forward(self, x):
        y = self.conv[self.task](x)

        return y


class SerieAdapterModule(nn.Module):
    
    def __init__(self, planes, nb_tasks=1):
        super(SerieAdapterModule, self).__init__()

        self.task = 0
        self.nb_task = nb_tasks
        self.alfa = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes)) for i in range(nb_tasks)])
    
    def set_index(self, index):
        self.task = index
        for i in range(self.nb_task):
            if i == index:
                m = self.alfa[i]
                m[0].bias.requires_grad = True
                m[0].weight.requires_grad = True
                m[1].weight.requires_grad = True
            else:
                m = self.alfa[i]
                m[0].bias.requires_grad = False
                m[0].weight.requires_grad = False
                m[1].weight.requires_grad = False

    def forward(self, x):

        y = x + self.alfa[self.task](x)
        
        return y


# No projection: identity shortcut
class BasicRebuffiBlock(nn.Module):
   
    def __init__(self, serie, in_planes, planes, stride=1, first=0, nb_tasks=1):
        super(BasicRebuffiBlock, self).__init__()
        self.serie = serie
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if serie:
            self.alfa1 = SerieAdapterModule(planes, nb_tasks)
        else:  # is parallel
            self.alfa1 = ParallelAdapterModule(in_planes, planes, stride, nb_tasks)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if serie:
            self.alfa2 = SerieAdapterModule(planes, nb_tasks)
        else:  # is parallel
            self.alfa2 = ParallelAdapterModule(planes, planes, nb_tasks=nb_tasks)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        
        self.first = first
        if first:
            self.downsample = nn.Sequential(
                  nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
    
        self.index = 0
        self.classes = nb_tasks
        
        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False
        if first:
            self.downsample[0].weight.requires_grad = False
    
    def set_index(self, index):
        if 0 <= index < self.classes:
            self.index = 0
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
        if self.serie:
            out = out + self.alfa1(out)
        else:
            out = out + self.alfa1(x)
        
        out = self.bn1[self.index](out)
        out = self.relu(out)
        
        x = out
        out = self.conv2(out)
        
        if self.serie:
            out = out + self.alfa2(out)
        else:
            out = out + self.alfa2(x)

        out = self.bn2[self.index](out)

        out += residual
        out = self.relu(out)
    
        return out


class RebuffiNet(nn.Module):
    def __init__(self, serie=True, layers=[2,2,2,2], classes=[10], fc=True):
        super(RebuffiNet, self).__init__()
        nb_tasks = len(classes)
        self.serie = serie
        self.models = nb_tasks
        self.in_planes = int(64)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(64) for i in range(self.models)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(int( 64), layers[0], nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(int(128), layers[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(int(256), layers[2], stride=2, nb_tasks=nb_tasks)
        self.layer4 = self._make_layer(int(512), layers[3], stride=2, nb_tasks=nb_tasks)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc_exists = fc
        if fc:
            self.fc = nn.ModuleList([nn.Linear(int(512), c) for c in classes])
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.task = 0
        self.set_index(self.task)

        self.conv1.weight.requires_grad = False
    
    def _make_layer(self, planes, nblocks, stride=1, nb_tasks=1):
        layers = [BasicRebuffiBlock(self.serie, self.in_planes, planes, stride, first=True, nb_tasks=nb_tasks)]
        self.in_planes = planes
        for i in range(1, nblocks):
            layers.append(BasicRebuffiBlock(self.serie, self.in_planes, planes, nb_tasks=nb_tasks))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def set_index(self, index):
        self.task = index
        
        for x in self.modules():
            if isinstance(x, SerieAdapterModule) \
                  or isinstance(x, ParallelAdapterModule) \
                  or isinstance(x, BasicRebuffiBlock):
                x.set_index(index)

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
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1[self.task](x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.fc_exists:
            x = self.fc[self.task](x)
        
        return x


def rebuffi_net18(model_classes, serie=True, pre_imagenet=True, pretrained=None, bn=True, fc=True):
    
    model = RebuffiNet(serie, layers=[2, 2, 2, 2], classes=model_classes, fc=fc)
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

    if not bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    print("Model pretrained loaded")
    return model

