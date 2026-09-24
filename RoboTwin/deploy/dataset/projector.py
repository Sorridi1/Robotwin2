import os
import numpy as np

from dataset.constants import *
from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot


class Projector:
    def __init__(self, calib_path=None):

        R_base_camera = np.array([
            [ 0.04640571,  0.49439835, -0.86799584],
            [ 0.99877564, -0.03787269,  0.03182584],
            [-0.01713869, -0.86841000, -0.49555054]
        ])

        t_base_camera = np.array([0.69952905, -0.07545494, 0.30516212])
        T_bc = np.eye(4)
        T_bc[:3, :3] = R_base_camera
        T_bc[:3, 3] = t_base_camera

        theta = np.deg2rad(90.0)
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]], float)

        R_bc_new = Rz @ T_bc[:3, :3]
        t_bc_new = Rz @ T_bc[:3, 3]

        T_base_camera = np.eye(4)
        T_base_camera[:3, :3] = R_bc_new
        T_base_camera[:3, 3] = t_bc_new
        # cam to base
        T_camera_base = np.linalg.inv(T_base_camera)

        # cam to base
        self.cam_to_base = {}
        self.cam_to_base['243222076209'] = T_camera_base

        
    def project_tcp_to_camera_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        # input()
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        # print("tcp_brfore",tcp)
        result=mat_to_xyz_rot(
            self.cam_to_base[cam] @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ), 
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
        # print("tcp_after",result)

        return result

    def project_tcp_to_base_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base[cam]) @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ),
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
