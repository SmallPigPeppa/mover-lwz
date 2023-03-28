import os
import cv2
import sys
sys.path.append('.')
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
    if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

from pare.models import SMPL
from pare.utils.renderer import Renderer
from pare.utils.vis_utils import show_3d_pose, color_vertices
from pare.core.config import get_hparams_defaults, SMPL_MODEL_DIR
from pare.dataset import BaseDataset, MixedDataset, FitsDict, LMDBDataset
from pare.utils.geometry import estimate_translation, convert_weak_perspective_to_perspective, \
    convert_perspective_to_weak_perspective


def main(dataset_name, is_train):
    cfg = get_hparams_defaults()

    cfg.DATASET.IMG_RES = 224
    cfg.DATASET.NOISE_FACTOR = 0  # 0.4
    cfg.DATASET.ROT_FACTOR = 0  # 30
    cfg.DATASET.SCALE_FACTOR = 0.0  # 0.25
    cfg.DATASET.CROP_PROB = 0.0  # 1.0
    cfg.DATASET.CROP_FACTOR = 0.0  # 0.5
    cfg.DATASET.FLIP_PROB = 0.0
    cfg.DATASET.USE_SYNTHETIC_OCCLUSION = False
    cfg.DATASET.OCC_AUG_DATASET = 'pascal'
    cfg.DATASET.USE_3D_CONF = False

    cfg.DATASET.NUM_IMAGES = -1

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
    )

    renderer = Renderer(
        focal_length=cfg.DATASET.FOCAL_LENGTH,
        img_res=cfg.DATASET.RENDER_RES,
        faces=smpl.faces,
    )

    ds = BaseDataset( # LMDBDataset
        cfg.DATASET,
        dataset=dataset_name,
        num_images=cfg.DATASET.NUM_IMAGES,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset=ds,
        shuffle=False,
        num_workers=4,
        batch_size=1,
        pin_memory=True,
    )

    start_time = time.time()
    total_start_time = time.time()

    dataloader_iter = iter(dataloader)

    faulty_counter = 0
    for batch_idx in tqdm(range(len(dataloader))):
        try:
            batch = next(dataloader_iter)
        except:
            print(f'An exception occurred while fetching object at {batch_idx}')
            faulty_counter += 1
            continue
        images = batch['disp_img']
        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        # print(f'**** total {batch_idx:03d}: {time.time()-start_time:.6f}s '
        #       f'load {batch["load_time"].mean()}s '
        #       f'proc {batch["proc_time"].mean()}s')

        start_time = time.time()

        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        gt_out = smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * cfg.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        try:
            # Estimate camera translation given the model joints and 2D keypoints
            # by minimizing a weighted least squares loss
            gt_cam_t = estimate_translation(
                gt_model_joints,
                gt_keypoints_2d_orig,
                focal_length=cfg.DATASET.FOCAL_LENGTH,
                img_size=cfg.DATASET.IMG_RES,
                use_all_joints=True,
            )
        except:
            print(f'Cam params cannot be estimated, keypoints are problematic at id: {batch_idx}')
            faulty_counter += 1
            # import IPython; IPython.embed(); exit(1)
            continue

        gt_cam = convert_perspective_to_weak_perspective(gt_cam_t)

        gt_cam_t = convert_weak_perspective_to_perspective(
            gt_cam,
            focal_length=cfg.DATASET.FOCAL_LENGTH,
            img_res=cfg.DATASET.RENDER_RES,
        )

        render_images = renderer.visualize_tb(
            gt_vertices,
            gt_cam_t,
            images,
            sideview=True,
            multi_sideview=True,
        )
        # breakpoint()
        render_images = render_images.cpu().numpy().transpose(1, 2, 0) * 255
        render_images = np.clip(render_images, 0, 255).astype(np.uint8)

        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * cfg.DATASET.RENDER_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        visible_kps = gt_keypoints_2d_orig[0,:,:][gt_keypoints_2d_orig[0,:,2] > 0]
        x1, y1, x2, y2 = min(visible_kps[:,0]), min(visible_kps[:,1]), max(visible_kps[:,0]), max(visible_kps[:,1])
        # draw bbox
        render_images = cv2.rectangle(render_images, (int(x1), int(y1)), (int(x2), int(y2)), (0,225,0), 2)

        # plt.figure(figsize=(20,6))
        # plt.imshow(render_images)
        # plt.show()

        save_dir = f'/ps/scratch/ps_shared/mkocabas/pare_results/eft_dataset_renders/{dataset_name}_bbox'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(save_dir, f'{dataset_name}_{batch["sample_idx"][0].item():06d}.jpg'),
            cv2.cvtColor(render_images, cv2.COLOR_BGR2RGB)
        )

    print(f'Number of failures: {faulty_counter}')
    total_time = time.time() - total_start_time
    print(f'Time spent {total_time:.1f} s.\n'
          f'Avg {total_time/len(dataloader):.3f} s/it')


if __name__ == '__main__':
    # with threadpool_limits(limits=1):
    main(dataset_name=sys.argv[1], is_train=int(sys.argv[2]))