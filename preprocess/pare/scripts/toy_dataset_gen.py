# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import pyrender
import trimesh
import numpy as np
from smplx import SMPL


from pare.core.config import SMPL_MODEL_DIR
from pare.core.constants import SMPLH_TO_SMPL
from pare.utils.mesh_viewer import MeshViewer
from pare.utils.one_euro_filter import OneEuroFilter

def render(vertices, faces, width=512, height=512,
           cam_center=256, focal_length=1000, cam_pose=None):
    registered_keys = dict()
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=(0.3, 0.3, 0.3)
    )

    pc = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=cam_center, cy=cam_center,
        znear=0.05, zfar=100.0
    )

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 3])
    scene.add(pc, pose=camera_pose, name='cam')

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    scene_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(scene_mesh)

    viewer = pyrender.Viewer(
        scene, use_raymond_lighting=True, viewport_size=(width, height),
        cull_faces=False, run_in_thread=False, registered_keys=registered_keys,
    )

if __name__ == '__main__':
    mv = MeshViewer(body_color=(0.8, 0.8, 0.8, 1.0))

    poses = np.load('data/amass/MPI_Limits/03099/lar1_poses.npz')['poses']
    poses = poses[:, SMPLH_TO_SMPL]

    smpl = SMPL(model_path=SMPL_MODEL_DIR)
    faces = smpl.faces

    for idx, pose in enumerate(poses[1:]):
        print(f'{idx}/{poses.shape[0]}')
        vertices = smpl(
            body_pose=torch.from_numpy(pose[3:]).unsqueeze(0).float(),
            global_orient=torch.from_numpy(pose[:3]).unsqueeze(0).float(),
        ).vertices

        vertices = vertices.detach().cpu().numpy().squeeze()

        render(vertices, faces)