import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

# Author: Massimo Mancini and Fabio Cermelli


# Piggyback implementation
class MaskedConv2d(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, mask=1):

        super(MaskedConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.threshold = 0.0
        self.mask = nn.ParameterList([Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
                                      for i in range(0, mask)])
        self.masks = mask
        self.index = 0
        
        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False
        self.reset_mask()

    def set_index(self, index):
        if 0 <= index < self.masks:
            self.index = index
            for i, ms in enumerate(self.mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True

    def reset_mask(self):
        for mask_p in self.mask:
            mask_p.data.fill_(0.01)

    def forward(self, x):
        binary_mask = self.mask[self.index].clone()
        binary_mask.data = (binary_mask.data > self.threshold).float()
        W = binary_mask*self.weight
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)


# Massimo Mancini's ECCV submission
class QuantizedConv2d(nn.modules.conv.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, mask=1):
        super(QuantizedConv2d, self).__init__(
              in_channels, out_channels, kernel_size, stride, padding, dilation
              , groups, bias)
        
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False
        
        self.masks = mask
        self.mask = nn.ParameterList([Parameter(
              torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)) for i in range(0, mask)])
              
        self.threshold = 0.0
        
        self.bias_mask = nn.ParameterList([Parameter(torch.FloatTensor(1)) for i in range(0, mask)])
        self.scale_mask = nn.ParameterList([Parameter(torch.Tensor(1).view(-1, 1, 1, 1)) for i in range(0, mask)])
        self.scale_mask2 = nn.ParameterList([Parameter(torch.Tensor(1).view(-1, 1, 1, 1)) for i in range(0, mask)])
        
        self.reset_mask()
        self.index = 0
    
    def reset_mask(self):
        for mask_p in self.mask:
            mask_p.data.uniform_(0.00001, 0.00002)
        for mask in self.bias_mask:
            mask.data.fill_(0.0)
        for mask in self.scale_mask:
            mask.data.fill_(0.0)
        for mask in self.scale_mask2:
            mask.data.fill_(0.0)

    def set_index(self, index):
        if 0 <= index < self.masks:
            self.index = index

            for i, ms in enumerate(self.mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True
            for i, ms in enumerate(self.bias_mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True
            for i, ms in enumerate(self.scale_mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True
            for i, ms in enumerate(self.scale_mask2.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True

    def forward(self, x):
        binary_addition_masks = self.mask[self.index].clone()
        binary_addition_masks.data = (binary_addition_masks.data > self.threshold).float()

        # W= W_pretrained + a*M + b*W*M + c # w = K0*w + K1*1 + K2*M + K3*W*M with K0 = 0
        w = self.weight \
            + self.scale_mask[self.index] * binary_addition_masks \
            + self.scale_mask2[self.index] * self.weight * binary_addition_masks \
            + self.bias_mask[self.index]
        
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class CombinedLayers(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, mask=1, order=[0,1,2]):

        super(CombinedLayers, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.threshold = 0.0
        self.masks = mask
        self.index = 0

        # we have one mask for each task
        self.mask = nn.ParameterList([Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
                                      for i in range(0, mask)])

        # task order[0] is independent from the others, order[1] depends on order[0], order[2] on 0,1 and so on
        self.alphas = nn.ParameterList([nn.Parameter(torch.eye(mask, require_grad=True)[i]) for i in range(mask)])
        for i in range(len(order)):
            for j in range(i+1, len(order)):
                self.alphas[order[i]][order[j]].require_grad = False

        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False
        self.reset_mask()

    def reset_mask(self):
        for mask_p in self.mask:
            mask_p.data.fill_(0.01)

    def set_index(self, index):
        if 0 <= index < self.masks:
            self.index = index
            for i, ms in enumerate(self.mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True

    def make_binary_mask(self):
        final_binary_mask = torch.zeros(self.mask[self.index].shape)
        for i in range(self.masks):
            binary_mask = self.mask[i].clone()
            binary_mask.data = self.alphas[self.index][i]*(binary_mask.data > self.threshold).float()
            final_binary_mask = final_binary_mask + binary_mask
        return final_binary_mask

    def forward(self, x):
        binary_mask = self.make_binary_masks()
        W = binary_mask*self.weight
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
