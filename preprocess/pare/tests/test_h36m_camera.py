import os
import cv2
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from pare.models import SMPL
from pare.utils.vis_utils import show_3d_pose
from pare.utils.geometry import estimate_translation, batch_rodrigues, batch_rot2aa
from pare.utils.renderer_realcam import RealCamRenderer
from pare.dataset.preprocess.h36m_train import load_cameras
from pare.core.config import get_hparams_defaults, SMPL_MODEL_DIR
from pare.dataset import BaseDataset, MixedDataset, FitsDict, LMDBDataset

def get_cam_parameters(imgname):
    cam_id = imgname.split('/')[-2].split('.')[-1]

    camera_ids = {
        '54138969': 1,
        '55011271': 2,
        '58860488': 3,
        '60457274': 4,
    }

    rcams = load_cameras()
    R, T, f, c, k, p, name = rcams[camera_ids[cam_id]]

    # import IPython; IPython.embed(); exit()

    cam = {
        'R': R,
        'T': T,
        'f': f,
        'c': c,
        'name': name,
    }

    return cam


def main(dataset_name, is_train):
    cfg = get_hparams_defaults()

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
    )

    cfg.DATASET.NOISE_FACTOR = 0 # 0.4
    cfg.DATASET.ROT_FACTOR = 0 # 30
    cfg.DATASET.SCALE_FACTOR = 0.0 # 0.25
    cfg.DATASET.CROP_PROB = 0.0 # 1.0
    cfg.DATASET.CROP_FACTOR = 0.0 # 0.5
    cfg.DATASET.FLIP_PROB = 0.0
    cfg.DATASET.USE_SYNTHETIC_OCCLUSION = False

    ds = BaseDataset( # LMDBDataset
        cfg.DATASET,
        dataset=dataset_name,
        num_images=-1,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset=ds,
        shuffle=False,
        num_workers=1,
        batch_size=1,
        pin_memory=True,
    )

    start_time = time.time()
    total_start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        images = batch['img']
        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        print(imgnames, batch['is_flipped'])

        cam = get_cam_parameters(imgname=imgnames[0])
        orig_img = cv2.cvtColor(cv2.imread(imgnames[0]), cv2.COLOR_BGR2RGB) / 255.0

        print('cam T:', cam['T'][:,0])
        print('cam f:', cam['f'][:, 0])
        print('center', batch['center'])
        print('scale', batch['scale'] * 200.)
        camera_pose = np.eye(4)
        # camera_pose[:3,:3] = cam['R']
        # camera_pose[:3, 3] = cam['T'][:,0]
        camera_pose[0, 3] = 0.0
        camera_pose[1, 3] = 0.0
        camera_pose[2, 3] = 4.9

        # gt_pose_rot = batch_rodrigues(gt_pose[:,:3]).reshape(-1,3,3)
        # go = gt_pose_rot @ torch.from_numpy(np.invert(cam['R'])).unsqueeze(0)
        # go = batch_rot2aa(go)

        print('camera_pose:\n', camera_pose)
        gt_out = smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        gt_vertices = gt_out.vertices[0].detach().cpu().numpy()

        renderer = RealCamRenderer(
            focal_length=cam['f'],
            img_res=(orig_img.shape[1], orig_img.shape[0]),
            camera_center=cam['c'],
            faces=smpl.faces,
        )

        rend_img = renderer(vertices=gt_vertices, image=orig_img, camera_pose=camera_pose)
        rend_img *= 255
        rend_img = np.clip(rend_img, 0, 255).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(rend_img)
        plt.show()

        if batch_idx == 100:
            break

        print(f'**** total {batch_idx:03d}: {time.time()-start_time:.6f}s '
              f'load {batch["load_time"].mean()}s '
              f'proc {batch["proc_time"].mean()}s')

        start_time = time.time()

        if False:
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

            gt_out = smpl(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3]
            )
            gt_model_joints = gt_out.joints
            gt_vertices = gt_out.vertices

            # show_3d_pose(kp_3d=batch['pose_3d'][0], dataset='smplcoco')

            # import IPython; IPython.embed(); exit(1)

            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = \
                0.5 * cfg.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            # Estimate camera translation given the model joints and 2D keypoints
            # by minimizing a weighted least squares loss
            gt_cam_t = estimate_translation(
                gt_model_joints,
                gt_keypoints_2d_orig,
                focal_length=cfg.DATASET.FOCAL_LENGTH,
                img_size=cfg.DATASET.IMG_RES,
            )

            # gt_cam_t = torch.from_numpy(np.array([[0, 0, 2 * 5000 / 224]]))


            render_images = renderer.visualize_tb(
                gt_vertices,
                gt_cam_t,
                images,
                gt_keypoints_2d,
                skeleton_type='spin',
                sideview=True,
            )
            # breakpoint()
            render_images = render_images.cpu().numpy().transpose(1, 2, 0) * 255
            render_images = np.clip(render_images, 0, 255).astype(np.uint8)
            plt.figure(figsize=(20,6))
            plt.imshow(render_images)
            plt.show()

            # save_dir = os.path.join('.temp_output')
            # os.makedirs(save_dir, exist_ok=True)
            # cv2.imwrite(
            #     os.path.join(save_dir, f'result_{batch_idx:05d}.jpg'),
            #     cv2.cvtColor(render_images, cv2.COLOR_BGR2RGB)
            # )

    total_time = time.time() - total_start_time
    print(f'Time spent {total_time:.1f} s.\n'
          f'Avg {total_time/len(dataloader):.3f} s/it')


if __name__ == '__main__':
    # with threadpool_limits(limits=1):
    main(dataset_name='h36m', is_train=1)