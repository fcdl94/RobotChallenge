import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.models import ResNet,DenseNet
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
import copy
import custom_layers
import torch.nn.functional as F


# Define the basic convolutional operation of the residual block
def conv3x3(in_planes, out_planes, stride=1, quantized=0, masked=False):
    if masked:
        return custom_layers.MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    elif quantized:
        return custom_layers.QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, masks=1)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# Define a residual block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, first=False, masked=False, bn=False, quantized=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, quantized=quantized, masked=masked)
	self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quantized=quantized, masked=masked)
	self.bn2 = nn.BatchNorm2d(planes)

        if first:
		if masked:
			self.downsample = nn.Sequential(custom_layers.MaskedConv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes))
		elif quantized:
			self.downsample = nn.Sequential(custom_layers.QuantizedConv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, masks=1),
                                        nn.BatchNorm2d(planes))
		else:
			self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes))
	else: 
		self.downsample = Identity()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out




# Function for an identity operation (dumb)
class Identity(nn.Module):

    def updateOutput(self, input):
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def clearState(self):
	clear(self, 'gradInput')

    def forward(self,x):
	return x



# Create the network
class ResNet28(nn.Module):
    def __init__(self, block, n_size, init=0.01,classes=1000, masked=False,quantized=0,bn=False):
        super(ResNet28, self).__init__()
        self.inplane = 16

	if masked:
        	self.conv1 = custom_layers.MaskedConv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.inplane)	
	elif quantized:
		self.conv1 =custom_layers.QuantizedConv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False, masks=1)
        	self.bn1 = nn.BatchNorm2d(self.inplane)
	else:
		self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        	self.bn1 = nn.BatchNorm2d(self.inplane)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, blocks=n_size, stride=2, masked=masked,quantized=quantized,bn=bn)
        self.layer2 = self._make_layer(block, 128, blocks=n_size, stride=2, masked=masked,quantized=quantized,bn=bn)
        self.layer3 = self._make_layer(block, 256, blocks=n_size, stride=2, masked=masked,quantized=quantized,bn=bn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
	
	self.fc = nn.Linear(256, classes)


    def _make_layer(self, block, planes, blocks, stride, masked=False,quantized=0,bn=False):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for i in range(len(strides)):
            layers.append(block(self.inplane, planes, strides[i], first= (i==0),masked=masked,quantized=quantized,bn=bn))
            self.inplane = planes

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
        x = self.fc(x)
        return x



# Initialize standard ResNet
def resnet28(pretrained=None, classes=1000, frozen=False):
	model = ResNet28(BasicBlock, 4, classes=classes)
	if pretrained:
		try:
			state = model.state_dict()
			state.update(torch.load(pretrained)['state_dict'])
			model.load_state_dict(state)
		except:
			model_dict=model.state_dict()
			state_dict =torch.load(pretrained)['state_dict']
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
   				name = k[7:] # remove `module.`
				if name in model_dict:
    					new_state_dict[name] = v
			state = model.state_dict()
			state.update(new_state_dict)
			model.load_state_dict(state)
	
	if frozen==1:
		for name,param in model.named_parameters():
			if "fc" in name:
				param.requires_grad=True
			else:
				param.requires_grad=False
    	return model


# Initialize masked models
def piggyback28(pretrained=None, classes=1000, masked=False,bn=False, quantized=False):
	model = ResNet28(BasicBlock, 4, classes=classes, masked=masked,quantized=quantized,bn=bn)
	if pretrained:
		if classes==1000:
			state_dict =torch.load(pretrained)['state_dict']
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
   				name = k[7:] # remove `module.`
    				new_state_dict[name] = v
			state = model.state_dict()
			state.update(new_state_dict)
			model.load_state_dict(state)
		else:
			try:
				model.load_state_dict(torch.load(pretrained)['state_dict'])
			except:
				model_dict=model.state_dict()
				state_dict =torch.load(pretrained)['state_dict']
				new_state_dict = OrderedDict()
				for k, v in state_dict.items():
   					name = k[7:] # remove `module.`
					if name in model_dict:
    						new_state_dict[name] = v
				state = model.state_dict()
				state.update(new_state_dict)
				model.load_state_dict(state)
	
	for name,param in model.named_parameters():
			if ("mask" in name) or ("fc" in name):
				param.requires_grad=True
			else:
				param.requires_grad=False

    	return model

