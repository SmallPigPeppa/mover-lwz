import h5py
import numpy as np

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

data_root = os.path.join('samples_clean_gta', 'FPS-5')
rec_idx = '2020-06-11-10-06-48'
info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
info_npz = np.load(open(os.path.join(data_root, rec_idx, 'info_frames.npz'), 'rb'))
kpts_npz = info_npz['joints_3d_world']
kpts_pkl = [i['kpvalue'] for i in info_pkl]

a = 1

# step1: gta_origin to smplx

# step3: save smplx as hdf5
