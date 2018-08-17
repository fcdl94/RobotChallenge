import torch.nn as nn
import torch
import math
from PoseEstimation.utils import rotation_equals, geodesic_distance
import torch.nn.functional as f

class PE3DLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self, classes):
        super(PE3DLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.GeodesicDistance = GeodesicDistance()  # this is the L2 norm
        self.classes = classes
    
    def forward(self, x, target):
        """ Input should be BS x C+4
            Target should be BS x 1+4 (class label + 4 Quaternion values)
        """
        class_input, rot_input = x[:, 0:self.classes], x[:, self.classes:]
        class_target, rot_target = target[:, 0], target[:, 1:]

        # Quaternion / quaternion_norm  (L2 regularization)
        rot_input = f.normalize(rot_input, p=2, dim=1)
        
        ce_loss = self.CrossEntropyLoss(class_input, class_target.long())
        distance = self.GeodesicDistance(rot_input, rot_target)
        final_loss = ce_loss + distance.mean()
        return final_loss


class PEMetric(nn.Module):

    def __init__(self, classes, threshold=10):
        """
        :param classes: The number of classes to be detected
        :param threshold: The max angle error to define that a pose is correct
        """
        super(PEMetric, self).__init__()
        self.classes = classes
        self.threshold = math.radians(threshold)
        
    def forward(self, x, target):
        """
            Input should be BS x C+4
            Target should be BS x 1+4 (class label + 4 Quaternion values)
        """
        
        class_input, rot_input = x[:, 0:self.classes], x[:, self.classes:]
        class_target, rot_target = target[:, 0].long(), target[:, 1:]
        
        # Quaternion / quaternion_norm  (L2 regularization)
        rot_input = f.normalize(rot_input, p=2, dim=1)
        
        pred = torch.max(class_input, 1)[1]
        correct_class = pred.eq(class_target.data.view_as(pred))
        
        correct_pose = rotation_equals(rot_input, rot_target, self.threshold)

        correct = (correct_class & correct_pose).cpu()  # to be correct both must be correct [1 and 1]
        
        return correct.sum(-1)


class GeodesicDistance(nn.Module):
    
    def __init__(self):
        super(GeodesicDistance, self).__init__()

    def forward(self, x, target):
        """ Input  should be BS x 4 (Quaternion as q = a + bi + cj + dk)
            Target should be BS x 4 (Quaternion as q = a + bi + cj + dk)
        """
        return geodesic_distance(x, target)
