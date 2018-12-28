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
