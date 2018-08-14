import torch
import torch.nn.functional as F
import torch.nn as nn

# Author: Massimo Mancini and Fabio Cermelli


# Piggyback implementation
class MaskedConv2d(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, model_size=1):

        super(MaskedConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.threshold = 0.0
        self.mask = nn.ParameterList([nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
                                      for i in range(0, model_size)])

        self.index = 0
        self.reset_mask()

    def set_index(self, index):
        if index >= 0:
            self.index = index
            self.weight.requires_grad = False
            if self.bias:
                self.bias.requires_grad = False
            for i, ms in enumerate(self.mask.parameters()):
                if not i == index:
                    ms.requires_grad = False

    def reset_mask(self):
        for mask_p in self.mask:
            mask_p.data.fill_(0.01)

    def forward(self, x):
        if self.index == 0:
            W = self.weight
        else:
            binary_mask = self.mask[self.index].clone()
            binary_mask.data = (binary_mask.data > self.threshold).float()
            W = binary_mask*self.weight
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
