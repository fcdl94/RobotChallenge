import math
import torch

def rot_matrix_to_RPY(matrix):
    if matrix.shape != (3, 3):
        return None
    
    teta = math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
    ct = math.cos(teta)
    if ct == 0:
        ct += 1e-5
    roll = math.atan2(matrix[2, 1]/ct, matrix[2, 2]/ct)
    yaw = math.atan2(matrix[1, 0]/ct, matrix[0, 0]/ct)
    
    return [roll, teta, yaw]

def rotation_equals(rot1, rot2, threshold):
    if rot1.size() != rot2.size():
        cor = torch.abs(rot1 - rot2).sum(-1) < threshold
        return cor
    else:
        return 0