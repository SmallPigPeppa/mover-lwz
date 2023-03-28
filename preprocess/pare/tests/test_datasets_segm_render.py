import os
import cv2
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import neural_renderer as nr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from pare.models import SMPL
from pare.utils.renderer import Renderer
from pare.utils.geometry import estimate_translation
from pare.utils.vis_utils import show_3d_pose, color_vertices
from pare.core.config import get_hparams_defaults, SMPL_MODEL_DIR
from pare.dataset import BaseDataset, MixedDataset, FitsDict, LMDBDataset
from pare.utils.image_utils import generate_part_labels, get_body_part_texture, get_default_camera


def main(dataset_name, is_train):
    cfg = get_hparams_defaults()

    cfg.DATASET.IMG_RES = 224
    cfg.DATASET.NOISE_FACTOR = 0  # 0.4
    cfg.DATASET.ROT_FACTOR = 0  # 30
    cfg.DATASET.SCALE_FACTOR = 0.0  # 0.25
    cfg.DATASET.CROP_PROB = 1.0  # 1.0
    cfg.DATASET.CROP_FACTOR = 0.5  # 0.5
    cfg.DATASET.FLIP_PROB = 0.0
    cfg.DATASET.USE_SYNTHETIC_OCCLUSION = False
    cfg.DATASET.OCC_AUG_DATASET = 'pascal'
    cfg.DATASET.USE_3D_CONF = True

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

    ###########################################
    neural_renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=cfg.DATASET.RENDER_RES,
        image_size=cfg.DATASET.RENDER_RES,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=False,
    )

    body_part_texture = get_body_part_texture(smpl.faces, n_vertices=6890)

    K, R = get_default_camera(focal_length=cfg.DATASET.FOCAL_LENGTH, img_size=cfg.DATASET.RENDER_RES)
    smpl_faces = torch.from_numpy(smpl.faces.astype(np.int32)).unsqueeze(0)

    n_parts = 24
    part_label_bins = (torch.arange(int(n_parts)) / float(n_parts) * 255.) + 1
    ###########################################

    start_time = time.time()
    total_start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        images = batch['img']
        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        print(imgnames[0])

        # if not 'flat_packBags_00/image_00151.jpg' in imgnames[0]:
        #     continue

        # if not batch_idx % 64 == 0:
        #     continue
        # if batch_idx == 100:
        #     break

        print(f'**** total {batch_idx:03d}: {time.time()-start_time:.6f}s '
              f'load {batch["load_time"].mean()}s '
              f'proc {batch["proc_time"].mean()}s')

        start_time = time.time()

        if True:
            images = batch['disp_img']
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

            gt_betas = torch.tensor([[0.5, 0.5, -0.5, 0.5, -0.35, 0, 0, 0.4, 0, 0.25]])
            gt_out = smpl(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3]
            )
            gt_model_joints = gt_out.joints
            gt_vertices = gt_out.vertices

            print('joint_conf:', batch['pose_3d'][0][:,-1])
            # import IPython; IPython.embed(); exit(1)
            show_3d_pose(kp_3d=batch['pose_3d'][0], dataset='common')

            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = \
                0.5 * cfg.DATASET.RENDER_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            # import IPython; IPython.embed(); exit(1)
            try:
                # Estimate camera translation given the model joints and 2D keypoints
                # by minimizing a weighted least squares loss
                gt_cam_t = estimate_translation(
                    gt_model_joints,
                    gt_keypoints_2d_orig,
                    focal_length=cfg.DATASET.FOCAL_LENGTH,
                    img_size=cfg.DATASET.RENDER_RES,
                    use_all_joints=True,
                )
            except:
                import IPython; IPython.embed(); exit(1)

            # gt_cam_t = torch.from_numpy(np.array([[0, 0, 2 * 5000 / 224]]))

            ###################################################################
            gt_segm_mask, gt_segm_rgb = generate_part_labels(
                vertices=gt_vertices.to('cuda').float(),
                faces=smpl_faces.to('cuda').float(),
                cam_t=gt_cam_t.to('cuda').float(),
                K=K.to('cuda').float(),
                R=R.to('cuda').float(),
                body_part_texture=body_part_texture.to('cuda').float(),
                neural_renderer=neural_renderer,
                part_bins=part_label_bins.to('cuda').float(),
            )

            gt_segm_mask = gt_segm_mask.detach().cpu().numpy()

            ##################################################################
            # import IPython; IPython.embed(); exit(1)
            # breakpoint()
            vert_color = []
            pose_conf = batch['pose_conf'].numpy()
            for pc in pose_conf:
                vert_color.append(color_vertices(pc)[None, ...])
            vert_color = np.vstack(vert_color)
            print('pose_conf:', pose_conf)
            render_images = renderer.visualize_tb(
                gt_vertices,
                gt_cam_t,
                images,
                gt_keypoints_2d,
                skeleton_type='spin',
                sideview=True,
                segm_masks=gt_segm_mask,
                mesh_filename='logs/figures/model/result.obj',
                # vertex_colors=vert_color
            )
            # breakpoint()
            render_images = render_images.cpu().numpy().transpose(1, 2, 0) * 255
            render_images = np.clip(render_images, 0, 255).astype(np.uint8)
            plt.figure(figsize=(20,6))
            plt.imshow(render_images)

            # import IPython; IPython.embed(); exit(1)

            plt.show()

            # save_dir = os.path.join('logs/figures/occ_aug_samples')
            # os.makedirs(save_dir, exist_ok=True)
            # cv2.imwrite(
            #     os.path.join(save_dir, f'result_{batch_idx:05d}_crop.jpg'),
            #     cv2.cvtColor(render_images, cv2.COLOR_BGR2RGB)
            # )

            if batch_idx == 50:
                exit()

    total_time = time.time() - total_start_time
    print(f'Time spent {total_time:.1f} s.\n'
          f'Avg {total_time/len(dataloader):.3f} s/it')


if __name__ == '__main__':
    # with threadpool_limits(limits=1):
    main(dataset_name=sys.argv[1], is_train=int(sys.argv[2]))