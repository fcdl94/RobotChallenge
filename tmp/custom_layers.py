import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np


# Piggyback implementation
class MaskedConv2d(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, th=0, masks=1):
        
	super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation
           ,groups, bias)
        
	self.weight.requires_grad=False

	if bias:
		self.bias.requires_grad=False
	self.threshold=0.0

	self.mask = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

	self.threshold_mask=0	
	self.reset_mask()

    def reset_mask(self):
        self.mask.data.fill_(0.01)

    def forward(self, input):
	binary_mask=self.mask.clone()
	binary_mask.data=(binary_mask.data>self.threshold).float()
	W = (binary_mask)*self.weight
        return F.conv2d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Our ECCV submission
class QuantizedConv2d(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, masks=1):
        
        super(QuantizedConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation
               ,groups, bias)
            
        self.weight.requires_grad=False
    
        self.n_masks=1
        self.additional_mask = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    
        if bias:
            self.bias.requires_grad=False
    
        self.threshold=0.0
    
        self.bias_mask=Parameter(torch.FloatTensor(1))
        self.scale_mask=Parameter(torch.Tensor(self.n_masks).view(-1,1,1,1))
        self.scale_mask2=Parameter(torch.Tensor(self.n_masks).view(-1,1,1,1))
    
        
        self.reset_mask()

    def reset_mask(self):
        self.additional_mask.data.uniform_(0.00001,0.00002)
        self.bias_mask.data.fill_(0.0)
        self.scale_mask.data.fill_(0.0)
        self.scale_mask2.data.fill_(0.0)


    def forward(self, input):

        binary_addition_masks=self.additional_mask.clone()
        binary_addition_masks.data=(binary_addition_masks.data>self.threshold).float()
    
        # W= W_pretrained + a*M + b*W*M + c
        W = self.weight+self.scale_mask*binary_addition_masks+self.scale_mask2*self.weight*binary_addition_masks+self.bias_mask
    
        return F.conv2d(input, W, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
