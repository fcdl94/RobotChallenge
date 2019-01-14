import math
import torch
import numpy as np
from math import sqrt


def quaternion_from_matrix(matrix, row=True):
    """Initialise from matrix representation
    Create a Quaternion by specifying the 3x3 rotation matrix
    (as a numpy array) from which the quaternion's rotation should be created.
    """
    try:
        shape = matrix.shape
    except AttributeError:
        raise TypeError("Invalid matrix type: Input must be a 3x3 numpy matrix")

    if shape == (3, 3):
        R = matrix
    else:
        raise ValueError("Invalid matrix shape: Input must be a 3x3 or 4x4 numpy array or matrix")

    # Check matrix properties
    if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), atol=1e-5):
        raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse: "
                         + str(np.dot(R, R.conj().transpose())))
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")

    # This method assumes row-vector and post-multiplication of that vector
    if row:
        m = matrix
    else:
        m = matrix.conj().transpose()
    
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q)
    q *= 0.5 / sqrt(t)
    # Normalize again, there can be some small numerical errors
    q = q / np.linalg.norm(q)
    return q


def geodesic_distance(q1, q2):
    val = torch.abs(torch.bmm(q1.view(q1.size()[0], 1, 4), q2.view(q2.size()[0], 4, 1)))
    for i, v in enumerate(val):
        if v > 1.0:
            val[i] = 1.0
    d = 2*torch.acos(val)
    for v in d:
        if math.isnan(v):
            raise Exception("NaN in geodesic distance! \nV:{} \nD:{}".format(val, d))
    return d


def rot_matrix_to_RPY(matrix):
    if matrix.shape != (3, 3):
        return None
    
    teta = math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))

    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    
    return [roll, teta, yaw]


def rotation_equals(rot1, rot2, threshold):
    cor = geodesic_distance(rot1, rot2) < threshold
    return cor.squeeze()



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

