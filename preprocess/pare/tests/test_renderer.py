import torch
import numpy as np
import pyrender
import trimesh
from pare.core.config import SMPL_MODEL_DIR
from pare.models import SMPL
import pickle as pkl
import joblib

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors


def add_joints(scene, joints, color=[0.9, 0.1, 0.1, 1.0]):
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)


def debug_joints(vertices, faces, smpl_joints):


    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')
    # vertex_colors = np.random.random(size=[vertices.shape[0], 4])
    print(np.unique(smpl_segmentation['smpl_index']))
    vertex_colors = np.ones((vertices.shape[0], 4)) * np.array([0.3, 0.3, 0.3, 1.])
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    # norm_gt = mpl_colors.Normalize(vmin=0, vmax=trunc_val)
    vertex_colors[:, :3] = cm(norm_gt(smpl_segmentation['smpl_index']))[:, :3]

    # vertex_colors[smpl_segmentation['smpl_index'] == 1, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 4, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 7, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 10, 3] = 0.0
    #
    # print(smpl_segmentation['part2num'])
    # breakpoint()

    # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(
        vertices,
        faces,
        vertex_colors=vertex_colors,
    )

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene(bg_color=(0,0,0))
    scene.add(mesh)

    # add_joints(scene, coco_joints, color=[1.0,0.0,0.0,1.0])
    add_joints(scene, smpl_joints, color=[1.0,0.0,0.0,1.0])

    pyrender.Viewer(scene, use_raymond_lighting=True)


def main():
    mean_pose = np.array([[ 0.,  0.,  0., -0.22387259,  0.0174436 ,
            0.09247071, -0.23784273, -0.04646965, -0.07860077,  0.27820579,
            0.01414277,  0.01381316,  0.43278152, -0.06290711, -0.09606631,
            0.50428283,  0.00345129,  0.0609754 ,  0.02297339, -0.03170039,
            0.00579749,  0.00695809,  0.13169473, -0.05443741, -0.05891175,
           -0.17524343,  0.13545137,  0.0134158 , -0.00365581,  0.00887857,
           -0.20932178,  0.16004365,  0.10919978, -0.03871734,  0.0823698 ,
           -0.20413892, -0.0056038 , -0.00751232, -0.00347825, -0.02369   ,
           -0.12479898, -0.27360466, -0.04594801,  0.19914683,  0.23728603,
            0.06672108, -0.04049612,  0.03286229,  0.05357843, -0.29137463,
           -0.69688406,  0.05585425,  0.28579422,  0.65245777,  0.12222859,
           -0.91159104,  0.23825037, -0.03660429,  0.92367181, -0.25544496,
           -0.06566227, -0.1044708 ,  0.05014435, -0.03878127,  0.09087035,
           -0.07071638, -0.14365816, -0.05897377, -0.18009904, -0.08745479,
            0.10929292,  0.20091476]])

    mean_shape = np.array([[ 0.20560974,  0.33556296, -0.35068284,  0.35612895,  0.41754073,
            0.03088791,  0.30475675,  0.23613405,  0.20912663,  0.31212645]])

    pose_params  = torch.zeros(1, 72) # torch.from_numpy(mean_pose).float() # torch.rand(1, 72) * 0.2
    shape_params = torch.zeros(1, 10) # torch.from_numpy(mean_shape).float() # torch.rand(1, 10) * 0.03

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )
    out = smpl(betas=shape_params, body_pose=pose_params[:,3:], global_orient=pose_params[:,:3])
    verts = out.vertices.cpu().detach().numpy()[0]
    joints3d = out.joints.cpu().detach().numpy()[0]
    smpl_joints = joints3d

    debug_joints(verts, smpl.faces, smpl_joints)

if __name__ == '__main__':
    main()
