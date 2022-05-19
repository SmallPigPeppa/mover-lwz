import torch
import trimesh
import joblib
import pyrender
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from smplx.body_models import SMPLX
import matplotlib.colors as mpl_colors

from pare.core.config import SMPLX_MODEL_DIR

KEYPOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    # smpl joints first 22
    'jaw',
    'left_eye_smplx',
    'right_eye_smplx',
    'left_index1', # smpl joint 23
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1', # smpl joint 24
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    # SMPL-X kinematic skeleton joints
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

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
    vertex_colors[:6890, :3] = cm(norm_gt(smpl_segmentation['smpl_index']))[:, :3]

    # vertex_colors[smpl_segmentation['smpl_index'] == 1, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 4, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 7, 3] = 0.0
    # vertex_colors[smpl_segmentation['smpl_index'] == 10, 3] = 0.0
    #
    # print(smpl_segmentation['part2num'])
    # breakpoint()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.6]
    tri_mesh = trimesh.Trimesh(
        vertices,
        faces,
        vertex_colors=vertex_colors,
    )

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene(bg_color=(255,255,255))
    scene.add(mesh)

    # add_joints(scene, coco_joints, color=[1.0,0.0,0.0,1.0])
    add_joints(scene, smpl_joints, color=[1.0,0.0,0.0,1.0])

    pyrender.Viewer(scene, use_raymond_lighting=True)


def main():
    smplx = SMPLX(model_path=SMPLX_MODEL_DIR)
    output = smplx()
    verts = output.vertices[0].detach().numpy()
    faces = smplx.faces
    smplx_joints = output.joints.cpu().detach().numpy()[0]

    # m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # m.show()
    print('Vertices: ', verts.shape)
    print('Joints: ', smplx_joints.shape)
    print('Joint names: ', len(KEYPOINT_NAMES))
    debug_joints(verts, faces, smplx_joints)


if __name__ == '__main__':
    main()
