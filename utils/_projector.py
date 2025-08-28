"""
_projector.py
"""
import os
import numpy as np

from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot

class Projector:
    def __init__(self, calib_path=None, T=None):
        if T is None:
            cam_T_base = np.load(os.path.join(calib_path, "Extrinsics.npy"), allow_pickle=True)
            cam_T_base = np.array(cam_T_base, dtype=np.float32).reshape(4, 4)
            self.base_T_cam = cam_T_base
            self.cam_T_base = np.linalg.inv(self.base_T_cam)
        else:
            self.cam_T_base = np.array(T, dtype=np.float32).reshape(4, 4)
            self.base_T_cam = np.linalg.inv(self.cam_T_base)

    # cam_T_tcp = cam_T_base * base_T_tcp
    def to_cam(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = 'ZXY'):
        base_T_tcp = xyz_rot_to_mat(
            tcp,
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )
        cam_T_tcp = self.cam_T_base @ base_T_tcp
        return mat_to_xyz_rot(
            cam_T_tcp,
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )

    # base_T_tcp = (cam_T_base)^-1 * cam_T_tcp
    def to_base(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = 'ZXY'):
        cam_T_tcp = xyz_rot_to_mat(
            tcp,
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )
        base_T_tcp = self.base_T_cam @ cam_T_tcp
        return mat_to_xyz_rot(
            base_T_tcp,
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )
