import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
