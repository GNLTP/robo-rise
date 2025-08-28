import numpy as np

from utils.constants import *

TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

# camera intrinsics
INTRINSICS = {
    "043322070878": np.array([[909.72656250, 0, 645.75042725, 0],
                              [0, 909.66497803, 349.66162109, 0],
                              [0, 0, 1, 0]]),
    "750612070851": np.array([[385.4960632324219, 0, 320.6309814453125, 0],
                              [0, 385.10809326171875, 236.02236938476562, 0],
                              [0, 0, 1, 0]]),
}

# inhand camera serial
INHAND_CAM = ["043322070878"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
INHAND_CAM_TCP = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0.077],
    [0, 0, 1, 0.2665],
    [0, 0, 0, 1]
])
