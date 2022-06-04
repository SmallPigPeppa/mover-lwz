import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm

from pare.core import constants
from pare.core import config as cfg
from pare.models.head.smpl_head import SMPL
from pare.utils.eval_utils import reconstruction_error


def eval_xuchen(dataset_path, results_dir):
    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], []

    # get a list of .pkl files in the directory
    files = []
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(dataset_path, 'sequenceFiles', split)
        files += [os.path.join(split_path, f)
                 for f in os.listdir(split_path) if f.endswith('.pkl')]

    device = 'cpu'
    smpl = SMPL(
        cfg.SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    ).to(device)

    J_regressor = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M)).float()
    joint_mapper_h36m = constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J14
    J_regressor_batch = J_regressor[None, :].expand(1, -1, -1)

    mpjpe, pampjpe = [], []
    # go through all the .pkl files
    for filename in tqdm(files):
        print(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        with open(os.path.join(results_dir, '/'.join(filename.split('/')[-2:])), 'rb') as f:
            result_data = pickle.load(f)

        pred_pose = result_data['thetas']
        pred_betas = result_data['betas']

        smpl_pose = data['poses']
        smpl_betas = data['betas']
        poses2d = data['poses2d']
        global_poses = data['cam_poses']
        genders = data['genders']
        valid = np.array(data['campose_valid']).astype(np.bool)
        num_people = len(smpl_pose)
        num_frames = len(smpl_pose[0])
        seq_name = str(data['sequence'])
        img_names = np.array(
            ['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
        # get through all the people in the sequence
        for i in range(num_people):
            valid_pred_pose = pred_pose[i][valid[i]]
            valid_pred_betas = pred_betas[i][valid[i]]


            valid_pose = smpl_pose[i][valid[i]]
            valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1), (num_frames, 1))
            valid_betas = valid_betas[valid[i]]
            valid_keypoints_2d = poses2d[i][valid[i]]
            valid_img_names = img_names[valid[i]]
            valid_global_poses = global_poses[valid[i]]
            gender = genders[i]
            # consider only valid frames
            for valid_i in range(valid_pose.shape[0]):
                part = valid_keypoints_2d[valid_i, :, :].T
                part = part[part[:, 2] > 0, :]
                bbox = [min(part[:, 0]), min(part[:, 1]),
                        max(part[:, 0]), max(part[:, 1])]
                center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

                # transform global pose
                pose = valid_pose[valid_i]
                extrinsics = valid_global_poses[valid_i][:3, :3]
                pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
                betas = valid_betas[valid_i]

                gpose = torch.from_numpy(pose).unsqueeze(0).float()
                gbetas = torch.from_numpy(betas).unsqueeze(0).float()

                ppose = torch.from_numpy(valid_pred_pose[valid_i]).unsqueeze(0).float()
                pbetas = torch.from_numpy(valid_pred_betas[valid_i]).unsqueeze(0).float()

                gt_vertices = smpl(
                    global_orient=gpose[:, :3],
                    body_pose=gpose[:, 3:],
                    betas=gbetas
                ).vertices
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

                pred_vertices = smpl(
                    global_orient=ppose[:, :3],
                    body_pose=ppose[:, 3:],
                    betas=pbetas
                ).vertices
                pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
                pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

                # Absolute error (MPJPE)
                error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

                # Reconstuction_error
                r_error, r_error_per_joint = reconstruction_error(
                    pred_keypoints_3d.cpu().numpy(),
                    gt_keypoints_3d.cpu().numpy(),
                    reduction=None,
                )

                mpjpe.append(error[0])
                pampjpe.append([r_error[0]])

                imgnames_.append(valid_img_names[valid_i])
                centers_.append(center)
                scales_.append(scale)
                poses_.append(pose)
                shapes_.append(betas)
                genders_.append(gender)

    mpjpe, pampjpe = np.array(mpjpe), np.array(pampjpe)
    print(mpjpe.mean(), pampjpe.mean())
    np.savez(
        'data/xu_chen_3dpw_results/results.npz',
        imgname=imgnames_,
        mpjpe=mpjpe,
        pampjpe=pampjpe
    )
    breakpoint()
    # store data
    # if not os.path.isdir(out_path):
    #     os.makedirs(out_path)
    # out_file = os.path.join(out_path,
    #                         '3dpw_all_test.npz')
    # np.savez(
    #     out_file,
    #     imgname=imgnames_,
    #     center=centers_,
    #     scale=scales_,
    #     pose=poses_,
    #     shape=shapes_,
    #     gender=genders_
    # )

if __name__ == '__main__':
    eval_xuchen(dataset_path=cfg.PW3D_ROOT, results_dir='data/xu_3dpw_results')