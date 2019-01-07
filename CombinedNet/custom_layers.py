import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


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

        self.order = order
        order_inverted = [order.index(i) for i in range(mask)]

        # task order[0] is independent from the others, order[1] depends on order[0], order[2] on 0,1 and so on
        # MUST BE INDEXED ON ORDER!
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros((i,1), requires_grad=True)) for i in order_inverted])

        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False
        self.reset()

    def reset(self):
        for mask_p in self.mask:
            mask_p.data.fill_(0.01)
        for alpha in self.alphas:
            alpha.data.fill_(0.001)

    def set_index(self, index):
        if 0 <= index < self.masks:
            self.index = index
            for i, ms in enumerate(self.mask.parameters()):
                if not i == index:
                    ms.requires_grad = False
                else:
                    ms.requires_grad = True
            for i, a in enumerate(self.alphas.parameters()):
                if not i == index:
                    a.requires_grad = False
                else:
                    a.requires_grad = True

    def make_binary_mask(self):

        final_binary_mask = (self.mask[self.index].clone().data > self.threshold).float()
        for i in range(len(self.alphas[self.index])):
            binary_mask = self.mask[self.order[i]].clone()
            binary_mask.data = (binary_mask.data > self.threshold).float()
            final_binary_mask += self.alphas[self.index][i] * binary_mask

        print(self.alphas[self.index])
        return final_binary_mask

    def forward(self, x):
        binary_mask = self.make_binary_mask()
        W = binary_mask*self.weight
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
