import h5py
import numpy
import numpy as np
from kpts_mapping.gta import GTA_ORIGINAL_NAMES, GTA_KEYPOINTS
from kpts_mapping.gta_im import GTA_IM_NPZ_KEYPOINTS, GTA_IM_PKL_KEYPOINTS
from kpts_mapping.smplx import SMPLX_KEYPOINTS
import pickle
import os


def create_mover_input(data_root,save_root, rec_idx):
    # step0:
    # merge info_frames.pickle info_frames.npz
    # trans world to camera
    info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
    info_npz = np.load(open(os.path.join(data_root, rec_idx, 'info_frames.npz'), 'rb'))
    kpts_npz = np.array(info_npz['joints_3d_world'])
    kpts_pkl = np.array([i['kpvalue'] for i in info_pkl]).reshape(kpts_npz.shape[0], -1, kpts_npz.shape[2])
    kpts_pkl_names = [i['kpname'] for i in info_pkl]
    world2cam = np.array(info_npz['world2cam_trans'])  # (N,4,4)
    kpts_world = np.concatenate((kpts_npz, kpts_pkl), axis=1)  # (N,119,3)
    print(f"#################process {os.path.join(data_root, rec_idx)}#####################")
    print("#################kpts_npz.shape, kpts_pkl.shape, kpts_world.shape########################\n", kpts_npz.shape,
          kpts_pkl.shape, kpts_world.shape)
    kpts_camera = []  # (N,119,3)
    for i in range(len(kpts_world)):
        r_i = world2cam[i][:3, :3].T
        t_i = world2cam[i][3, :3]
        kpts_camera_i = []  # (119,3)
        for j, kpt in enumerate(kpts_world[i]):
            if np.array_equal(kpt, [0, 0, 0]):
                kpts_camera_i.append(kpt)
            else:
                cam_point = np.matmul(r_i, kpt) + t_i
                kpts_camera_i.append(cam_point)
        kpts_camera.append(kpts_camera_i)

    # step1
    # gta_im to gta
    kpts_gta_im = numpy.array(kpts_camera)
    kpts_gta = numpy.zeros(shape=(len(kpts_gta_im), len(GTA_KEYPOINTS), 3))
    gta_im_names = GTA_IM_NPZ_KEYPOINTS + GTA_IM_PKL_KEYPOINTS
    mapping_list = []

    for i in range(len(kpts_gta)):
        mapping_list_i = []
        gta_im_names = GTA_IM_NPZ_KEYPOINTS + kpts_pkl_names[i]
        for kpt_name in GTA_ORIGINAL_NAMES:
            if kpt_name not in gta_im_names:
                mapping_list_i.append(-1)
            else:
                mapping_list_i.append(gta_im_names.index(kpt_name))
        mapping_list.append(mapping_list_i)

    for i in range(len(kpts_gta)):
        for j in range(len(kpts_gta[0])):
            if mapping_list[i][j] != -1:
                kpts_gta[i][j] = kpts_gta_im[i][mapping_list[i][j]]

    # average for nose
    # for i in range(len(kpts_gta)):
    #     flag = False
    #     for j in range(45, 51):
    #         if np.array_equal(kpts_gta[i][j], [0, 0, 0]):
    #             flag = True
    #             break
    #     if flag:
    #         kpts_gta[i][-1] = [0, 0, 0]
    #     else:
    #         kpts_gta[i][-1] = np.average(kpts_gta[i][45:51], axis=0)

    # step2
    # gta to smplx
    mapping_list2 = []
    valid_len = 0
    for kpt_name in SMPLX_KEYPOINTS:
        if kpt_name not in GTA_KEYPOINTS:
            mapping_list2.append(-1)
        else:
            mapping_list2.append(GTA_KEYPOINTS.index(kpt_name))
            valid_len += 1
    kpts_smplx = numpy.zeros(shape=(len(kpts_gta), len(SMPLX_KEYPOINTS), 3))
    for i in range(len(kpts_smplx)):
        for j in range(len(kpts_smplx[0])):
            if mapping_list2[j] != -1:
                kpts_smplx[i][j] = kpts_gta[i][mapping_list2[j]]

    # step3: save smplx as hdf5
    # with h5py.File(os.path.join(data_root, rec_idx, 'mover_input.hdf5'), 'w') as f:
    #     f.create_dataset('skeleton_joints', data=kpts_smplx)
    with h5py.File(os.path.join(save_root, rec_idx, 'mover_input.hdf5'), 'w') as f:
        f.create_dataset('skeleton_joints', data=kpts_smplx)


if __name__ == "__main__":

    # data_root = os.path.join('samples_clean_gta', 'FPS-5')
    data_root = '/mnt/mmtech01/dataset/vision_text_pretrain/gta-im/FPS-5'
    save_root = '/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/mover-input/FPS-5'

    # Get all the rec_idx values
    rec_idx_list = os.listdir(data_root)

    # Process each rec_idx
    for rec_idx in rec_idx_list:
        # Create the directory if it doesn't exist
        if not os.path.exists(os.path.join(save_root, rec_idx)):
            os.makedirs(os.path.join(save_root, rec_idx))
        # Call the create_mover_input function
        create_mover_input(data_root, save_root, rec_idx)

