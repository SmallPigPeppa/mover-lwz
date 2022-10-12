import h5py
import numpy
import numpy as np
from kpts_mapping.gta import GTA_ORIGINAL_NAMES, GTA_KEYPOINTS
from kpts_mapping.gta_im import GTA_IM_NPZ_KEYPOINTS, GTA_IM_PKL_KEYPOINTS
from kpts_mapping.smplx import SMPLX_KEYPOINTS
# # temp_name = "samples_clean/3_3_78_Female2_0.hdf5"
# gta_name = "samples_clean_gta/FPS-5-clean/FPS-5-2020-06-11-10-06-48.hdf5"
# info_npz_name = "samples_clean_gta/FPS-5/2020-06-11-10-06-48/info_frames.npz"
# # info_npz = np.load(open(info_npz_name, 'rb'))
# info_npz = np.load(info_npz_name)
# print(info_npz['joints_3d_world'].shape)
# joints = info_npz['joints_3d_cam'][:, :, [2, 0, 1]]
# # joints[:, :, 0] *=  -1
# # joints = info_npz['joints_3d_camera']
# with h5py.File(gta_name, "r+") as f:
#     print("Keys: %s" % f.keys())
#     del f['skeleton_joints']
#     print("Keys: %s" % f.keys())
#     f.create_dataset('skeleton_joints', data=joints)
#     print(f['skeleton_joints'])
#     print("Keys: %s" % f.keys())

# with h5py.File(gta_name, "r") as f:
#     print("Keys: %s" % f.keys())


# step0: info_frames.pickle to gta_origin
import pickle
import os

# step0:
# merge info_frames.pickle info_frames.npz
# trans world to camera
data_root = os.path.join('samples_clean_gta', 'FPS-5')
rec_idx = '2020-06-11-10-06-48'
info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
info_npz = np.load(open(os.path.join(data_root, rec_idx, 'info_frames.npz'), 'rb'))
kpts_npz = np.array(info_npz['joints_3d_world'])
kpts_pkl = np.array([i['kpvalue'] for i in info_pkl]).reshape(kpts_npz.shape[0], -1, kpts_npz.shape[2])
world2cam = np.array(info_npz['world2cam_trans'])
kpts_world = np.concatenate((kpts_npz, kpts_pkl), axis=1)
print(kpts_npz.shape, kpts_pkl.shape, kpts_world.shape)
kpts_camera = []
for i in range(len(kpts_world)):
    r_i = world2cam[i][:3, :3].T
    t_i = world2cam[i][3, :3]
    # cam_point_i = r_i * kpts_world[i] + t_i
    cam_point_i = [np.matmul(r_i, kpt) + t_i for kpt in kpts_world[i]]
    kpts_camera.append(cam_point_i)

# step1
# gta_im to gta
kpts_gta_im = numpy.array(kpts_camera)
kpts_gta = numpy.zeros(shape=(len(kpts_gta_im), len(GTA_KEYPOINTS), 3))
gta_im_names = GTA_IM_NPZ_KEYPOINTS + GTA_IM_PKL_KEYPOINTS
mapping_list=[]
for kpt_name in GTA_ORIGINAL_NAMES:
    if kpt_name not in gta_im_names:
        mapping_list.append(-1)
    else:
        mapping_list.append(gta_im_names.index(kpt_name))

for i in range(len(kpts_gta)):
    for j in range(len(kpts_gta[0])):
        if mapping_list[j]!=-1:
            kpts_gta[i][j] = kpts_gta_im[i][mapping_list[j]]
# print(mapping_list)

# average for nose
for i in range(len(kpts_gta)):
    kpts_gta[i][-1] = np.average(kpts_gta[i][45:51],axis=0)
# print(kpts_gta[0][45:51])
# print(kpts_gta[0])
# step2
# gta to smplx
mapping_list2=[]
valid_len=0
for kpt_name in SMPLX_KEYPOINTS:
    if kpt_name not in GTA_KEYPOINTS:
        mapping_list2.append(-1)
    else:
        mapping_list2.append(GTA_KEYPOINTS.index(kpt_name))
        valid_len+=1
print(len(mapping_list2))
print(mapping_list2)
kpts_smplx = numpy.zeros(shape=(len(kpts_gta), len(SMPLX_KEYPOINTS), 3))
for i in range(len(kpts_smplx)):
    for j in range(len(kpts_smplx[0])):
        if mapping_list2[j]!=-1:
            kpts_smplx[i][j] = kpts_gta[i][mapping_list2[j]]
print(kpts_smplx[0])
print(valid_len)

# print(kpts_camera[0][1])
# print(kpts_world[0][1])
# print(np.array(info_npz['joints_3d_cam'][0][1]))

# step1: gta_origin to smplx

# step3: save smplx as hdf5
# temp_name = "samples_clean/3_3_78_Female2_0.hdf5"
hdf5_name = "samples_clean_gta/FPS-5-clean/FPS-5-2020-06-11-10-06-48.hdf5"
# joints[:, :, 0] *=  -1
# joints = info_npz['joints_3d_camera']
with h5py.File(hdf5_name, "r+") as f:
    print("Keys: %s" % f.keys())
    del f['skeleton_joints']
    print("Keys: %s" % f.keys())
    f.create_dataset('skeleton_joints', data=kpts_smplx)
    print(f['skeleton_joints'])
    print("Keys: %s" % f.keys())

mapping_list3=[]
for idx,valid in enumerate(mapping_list2):
    if valid!=-1:
        mapping_list3.append(idx)
    else:
        mapping_list3.append(-1)
print(mapping_list3)

body_list=[]
from kpts_mapping.smplx import SMPLX_LIMBS
for i in SMPLX_LIMBS['body']:
    for j in i:
        if SMPLX_KEYPOINTS.index(j) not in body_list:
            body_list.append(SMPLX_KEYPOINTS.index(j))
print(body_list)

left_hand_list=[]
for i in SMPLX_LIMBS['left_hand']:
    for j in i:
        if SMPLX_KEYPOINTS.index(j) not in left_hand_list:
            left_hand_list.append(SMPLX_KEYPOINTS.index(j))
print(left_hand_list)


right_hand_list=[]
for i in SMPLX_LIMBS['right_hand']:
    for j in i:
        if SMPLX_KEYPOINTS.index(j) not in right_hand_list:
            right_hand_list.append(SMPLX_KEYPOINTS.index(j))
print(right_hand_list)