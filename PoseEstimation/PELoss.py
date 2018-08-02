import torch.nn as nn
import torch
from PoseEstimation.utils import rotation_equals

class PE3DLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self, classes):
        super(PE3DLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.PairwiseDistance = nn.PairwiseDistance(p=2)
        self.classes = classes
    
    def forward(self, input, target):
        """ Input should be BS x C+3
            Target should be BS x 1+3 (class label + 3 (RPY) values)
        """
        class_input, rot_input = input[:, 0:self.classes], input[:, self.classes:]
        class_target, rot_target = target[:, 0], target[:, 1:]
        
        ce_loss = self.CrossEntropyLoss(class_input, class_target.long())
        distance = self.PairwiseDistance(rot_input, rot_target)
        final_loss = ce_loss + distance.mean()
        return final_loss


class PEMetric(nn.Module):

    def __init__(self, classes, threshold):
        super(PEMetric, self).__init__()
        self.classes = classes
        self.threshold = threshold
        
    def forward(self, input, target):
        """ Input should be BS x C+3
            Target should be BS x 1+3 (class label + 3 (RPY) values)
        """
        class_input, rot_input = input[:, 0:self.classes], input[:, self.classes:]
        class_target, rot_target = target[:, 0].long(), target[:, 1:]
        
        pred = torch.max(class_input, 1)[1]
        correct_class = pred.eq(class_target.data.view_as(pred)).cpu()
        
        correct_pose = rotation_equals(rot_input, rot_target, self.threshold)

        correct = (correct_class + correct_pose) == 2  # to be correct both must be correct [1 and 1]
        
        return correct
