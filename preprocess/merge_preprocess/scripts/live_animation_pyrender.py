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
import numpy as np
from smplx import SMPL

from pare.core.config import SMPL_MODEL_DIR
from pare.core.constants import SMPLH_TO_SMPL
from pare.utils.mesh_viewer import MeshViewer
from pare.utils.one_euro_filter import OneEuroFilter

if __name__ == '__main__':
    mv = MeshViewer(body_color=(0.8, 0.8, 0.8, 1.0))

    poses = np.load('data/amass/MPI_Limits/03099/lar1_poses.npz')['poses']
    poses = poses[:, SMPLH_TO_SMPL]

    # Add random noise
    poses += (np.random.random(poses.shape) - 1.) / 75.

    min_cutoff = 0.004
    beta = 0.7

    one_euro_filter = OneEuroFilter(
        np.zeros_like(poses[0]), poses[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    smpl = SMPL(model_path=SMPL_MODEL_DIR)
    faces = smpl.faces

    for idx, pose in enumerate(poses[1:]):
        t = np.ones_like(pose) * (idx+1)
        pose = one_euro_filter(t, pose)

        print(f'{idx}/{poses.shape[0]}')
        vertices = smpl(
            body_pose=torch.from_numpy(pose[3:]).unsqueeze(0).float(),
            global_orient=torch.from_numpy(pose[:3]).unsqueeze(0).float(),
        ).vertices

        vertices = vertices.detach().cpu().numpy().squeeze()

        mv.update_mesh(vertices, faces)

    mv.close_viewer()