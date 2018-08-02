import math
import torch


def rot_matrix_to_RPY(matrix):
    if matrix.shape != (3, 3):
        return None
    
    teta = math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))

    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    
    return [roll, teta, yaw]


def rotation_equals(rot1, rot2, threshold):
    if rot1.size() == rot2.size():
        cor = torch.abs(rot1 - rot2).sum(-1) < threshold
        return cor
    else:
        return 0


def sanity_check_for_rot_matrix_to_RPY():
    # Sanity check for function to compute RPY
    
    import math
    import numpy as np
    
    # customize here value (keeps in radiants please [0, 2*math.pi])
    roll = math.pi
    pitch = 0.4506
    yaw = 1.1684
    
    print("roll = ", roll)
    print("pitch = ", pitch)
    print("yaw = ", yaw)
    print("")
    
    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    R = yawMatrix * pitchMatrix * rollMatrix
    print(R)
    
    print(rot_matrix_to_RPY(R))

