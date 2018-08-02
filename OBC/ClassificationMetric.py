import torch.nn as nn
import torch


class ClassificationMetric(nn.Module):
    
    def __init__(self):
        super(ClassificationMetric, self).__init__()
    
    def forward(self, input, target):
        """ Input should be BS x C
            Target should be BS x 1
        """
        pred = torch.max(input, 1)[1]
        correct_class = pred.eq(target.data.view_as(pred)).cpu()
        
        return correct_class.sum()
