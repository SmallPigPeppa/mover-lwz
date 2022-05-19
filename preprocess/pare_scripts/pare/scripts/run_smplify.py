import os
import sys
sys.path.append('.')
import torch
import pprint
import argparse
import numpy as np
from tqdm import tqdm
import skimage.io as io
from loguru import logger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pare.models import HMR, PARE
from pare.models import SMPL
from pare.dataset import BaseDataset
from pare.core import constants, config
from pare.utils.renderer import Renderer
from pare.smplify_cam.smplify import SMPLify
from pare.utils.eval_utils import reconstruction_error
from pare.utils.train_utils import load_pretrained_model
from pare.core.config import get_hparams_defaults, update_hparams, SMPL_MODEL_DIR
from pare.utils.geometry import rotation_matrix_to_angle_axis, convert_weak_perspective_to_perspective

# HMR_CKPT = 'logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/tb_logs_pare/0_af6934cb2bdf49bc892a61c0306526a9/checkpoints/epoch=77.ckpt'
# PARE_CKPT = 'logs/pare/25.10-pare_part_baseline_all_data_fix/25-10-2020_20-08-15_25.10-pare_part_baseline_all_data_fix_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train/epoch=14.ckpt.backup'
# PARE_CFG = 'logs/pare/25.10-pare_part_baseline_all_data_fix/25-10-2020_20-08-15_25.10-pare_part_baseline_all_data_fix_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train/config_to_run.yaml'

HMR_CKPT = '/ps/scratch/ps_shared/mkocabas/pare_results/spin_pretrained_ckpt_for_eft/epoch=77.ckpt'
PARE_CKPT = '/ps/scratch/ps_shared/mkocabas/pare_results/pare_pretrained_ckpt_for_smplify/epoch=14.ckpt.backup'
PARE_CFG = '/ps/scratch/ps_shared/mkocabas/pare_results/pare_pretrained_ckpt_for_smplify/config_to_run.yaml'

def benchmark(logs_1, logs_2):
    mpjpe_1 = np.load(os.path.join(logs_1, 'results.npz'))['pa_mpjpe']
    mpjpe_2 = np.load(os.path.join(logs_2, 'results.npz'))['pa_mpjpe']

    f1 = os.path.join(logs_1, 'images')
    f2 = os.path.join(logs_2, 'images')
    diff = mpjpe_1 - mpjpe_2

    c = 0
    for i, idx in enumerate(np.argsort(diff)[::-1][:200]):
        print(f'{i:03d} / 200')
        ai = io.imread(f'{f1}/{idx:06d}.png')
        bi = io.imread(f'{f2}/{idx:06d}.png')
        bi_crop = bi[bi.shape[0] // 2:,:,:]

        img = np.vstack([ai, bi_crop])

        plt.figure(figsize=(10, 10))
        plt.title(f'Error #1 {mpjpe_1[idx]:.2f} Error #2 {mpjpe_2[idx]:.2f}\ndiff {diff[idx]:.2f}')
        plt.imshow(img)
        plt.show()
        plt.close()

        inp = input('press "s" for save: ')

        if inp == 's':
            c += 1
            io.imsave(f'/home/mkocabas/Documents/PARE-CVPR2021/figures/smplify_focal_{c:02d}.png', img)


def main(args):
    cfg = get_hparams_defaults()

    if not args.use_hmr:
        cfg = update_hparams(PARE_CFG)

    logdir = os.path.join(
        args.logdir, args.dataset,
        f'fl-{args.focal_length}_camiter-{args.cam_opt_iter}_'
        f'poseiter-{args.pose_opt_iter}_params-{args.cam_opt_params}_'
        f'camlr-{args.cam_step_size}_batchsize-{args.batch_size}_'
        f'usehmr-{args.use_hmr}'
    )

    if args.log:
        os.makedirs(logdir, exist_ok=True)

        logger.add(
            os.path.join(logdir, 'run_smplify.log'),
            level='INFO',
            colorize=False,
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))

    batch_size = args.batch_size

    cfg.DATASET.NOISE_FACTOR = 0  # 0.4
    cfg.DATASET.ROT_FACTOR = 0  # 30
    cfg.DATASET.SCALE_FACTOR = 0.0  # 0.25
    cfg.DATASET.CROP_PROB = 0.0  # 1.0
    cfg.DATASET.CROP_FACTOR = 0.0  # 0.5
    cfg.DATASET.USE_SYNTHETIC_OCCLUSION = False

    smplify_focal_length = args.focal_length
    cam_num_iter = args.cam_opt_iter
    pose_num_iter = args.pose_opt_iter

    cam_opt_params = args.cam_opt_params # ('camera_translation', 'global_orient')
    # cam_opt_params = ('global_orient', 'camera_translation', 'focal_length', 'camera_rotation')

    smplify = SMPLify(
        cam_step_size=args.cam_step_size,
        pose_step_size=args.pose_step_size,
        batch_size=cfg.DATASET.BATCH_SIZE,
        pose_num_iters=pose_num_iter, # cfg.TRAINING.NUM_SMPLIFY_ITERS,
        focal_length=smplify_focal_length,
        camera_opt_params=cam_opt_params,
        use_weak_perspective=True,
        use_all_joints_for_camera=True,
        optimize_cam_only=False,
        cam_num_iters=cam_num_iter,
    )

    # import IPython; IPython.embed(); exit()
    logger.info(f'SMPLify running configuration: \n {pprint.pformat(args.__dict__)}')

    if args.use_hmr:
        model = HMR().to(device)
        model.eval()

        logger.info(f'Loading pretrained model from {HMR_CKPT}')
        ckpt = torch.load(HMR_CKPT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
    else:
        model_cfg = cfg
        model = PARE(
            backbone=model_cfg.PARE.BACKBONE,
            num_joints=model_cfg.PARE.NUM_JOINTS,
            softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
            num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
            focal_length=model_cfg.DATASET.FOCAL_LENGTH,
            img_res=model_cfg.DATASET.IMG_RES,
            pretrained=None,
            iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
            num_iterations=model_cfg.PARE.NUM_ITERATIONS,
            iter_residual=model_cfg.PARE.ITER_RESIDUAL,
            shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
            pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
            pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
            shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
            pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
            shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
            use_keypoint_features_for_smpl_regression=model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
            use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
            use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
            use_postconv_keypoint_attention=model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
            use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
            use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
            use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
            use_coattention=model_cfg.PARE.USE_COATTENTION,
            num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
            coattention_conv=model_cfg.PARE.COATTENTION_CONV,
            use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
            deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
            use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
            num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
            branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
        ).to(device)
        model.eval()

        logger.info(f'Loading pretrained model from {PARE_CKPT}')
        ckpt = torch.load(PARE_CKPT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    ).to(device)

    renderer = Renderer(
        focal_length=cfg.DATASET.FOCAL_LENGTH,
        img_res=cfg.DATASET.IMG_RES,
        faces=smpl.faces,
    )

    ds = BaseDataset(
        cfg.DATASET,
        dataset=args.dataset,
        num_images=-1,
        is_train=args.is_train,
    )

    dataloader = DataLoader(
        dataset=ds,
        shuffle=False,
        num_workers=batch_size // 8,
        batch_size=batch_size,
    )

    # Store SMPL parameters
    smpl_pose = np.zeros((len(ds), 72))
    smpl_betas = np.zeros((len(ds), 10))
    smpl_camera = np.zeros((len(ds), 3))
    nj = 17 if args.dataset == 'mpi-inf-3dhp' else 14
    pred_joints = np.zeros((len(ds), nj, 3))

    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float().to(device)

    errors, r_errors = [], []

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if batch_idx % args.skip != 0:
            continue
        print(f'Processing {batch_idx} / {len(dataloader)}')

        images = batch['img'].to(device)
        gt_keypoints_2d = batch['keypoints'].to(device)

        curr_batch_size = gt_keypoints_2d.shape[0]
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * cfg.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        pred = model(images)

        pred_rotmat = pred['pred_pose']
        pred_betas = pred['pred_shape']
        pred_cam_t = pred['pred_cam_t']
        weak_cam = pred['pred_cam']
        # Convert predicted rotation matrices to axis-angle
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach().view(-1, 3, 3)).reshape(curr_batch_size, -1)
        pred_pose[torch.isnan(pred_pose)] = 0.0

        # Run SMPLify optimization starting from the network prediction
        smplify_output = smplify(
            pred_pose.detach(), pred_betas.detach(),
            weak_cam.detach(),
            0.5 * cfg.DATASET.IMG_RES * torch.ones(curr_batch_size, 2, device=device),
            gt_keypoints_2d_orig
        )
        # logger.debug(f'Old camera parameters: {pred_cam_t.detach().cpu().numpy().tolist()}')

        new_opt_vertices, new_opt_joints, \
        new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss, new_focal_length, new_camera_rotation = smplify_output

        ################### MEASURE THE ERROR ############################
        joint_mapper_h36m = constants.H36M_TO_J17 if args.dataset == 'mpi-inf-3dhp' \
            else constants.H36M_TO_J14
        # joint_mapper_gt = constants.J24_TO_J17 if args.dataset == 'mpi-inf-3dhp' \
        #     else constants.J24_TO_J14
        J_regressor_batch = J_regressor[None, :].expand(new_opt_vertices.shape[0], -1, -1)

        gt_keypoints_3d = batch['pose_3d'].cuda()
        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, new_opt_vertices)
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
        error *= 1000
        r_error *= 1000

        errors += error.tolist()
        r_errors += r_error.tolist()
        ##################################################################
        # if 'focal_length' in cam_opt_params:
        #     new_focal_length = new_focal_length.detach().cpu().numpy()
        #
        #     gt_focal_length = []
        #
        #     for im_id, imgname in enumerate(batch['imgname']):
        #         sid = int(imgname.split('/')[-3][-1])
        #         gt_fl = 1500 if sid < 5 else 950
        #         height = batch['scale'][im_id] * 200.
        #         ratio = height / float(cfg.DATASET.IMG_RES)
        #         # ratio = 2048/scale if sid < 5 else 1080/scale
        #         gt_fl /= ratio
        #         gt_focal_length.append(gt_fl)
        #
        #     gt_focal_length = np.array(gt_focal_length)
        #     f_diff = np.abs(new_focal_length - gt_focal_length)
        #     logger.info(f'Focal L diff: {f_diff.mean():.2f}\u00B1{f_diff.std():.2f}, gt_f: {gt_focal_length.mean()}'
        #                 f'pred_f: {new_focal_length.mean()}')
        # import IPython; IPython.embed(); exit()
        # logger.debug(f'New camera parameters: {new_opt_cam_t.detach().cpu().numpy().tolist()}')

        logger.info(f'MPJPE: {np.array(errors).mean():.2f}\u00B1{np.array(errors).std():.2f}, '
                    f'PA-MPJPE: {np.array(r_errors).mean():.2f}\u00B1{np.array(r_errors).std():.2f}')

        new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

        # Save results
        start_idx = batch_idx * batch_size
        end_idx = start_idx + curr_batch_size
        smpl_pose[start_idx:end_idx, :] = pred_pose.detach().cpu().numpy()
        smpl_betas[start_idx:end_idx, :] = pred_betas.detach().cpu().numpy()
        smpl_camera[start_idx:end_idx, :] = pred_cam_t.detach().cpu().numpy()
        pred_joints[start_idx:end_idx, :, :] = pred_keypoints_3d.detach().cpu().numpy()

        ############### VISUALIZE ###############
        if args.show or args.save:
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

            for sidx in range(args.save_freq):
                # print(new_camera_rotation[sidx])

                renderer._set_focal_length(focal_length=cfg.DATASET.FOCAL_LENGTH)
                hmr_pred = renderer.visualize_tb(
                    vertices=pred['smpl_vertices'].detach()[sidx:sidx+1],
                    camera_translation=pred['pred_cam_t'].detach()[sidx:sidx+1],
                    images=images[sidx:sidx+1],
                    sideview=True,
                ).detach().cpu().numpy().transpose(1, 2, 0) * 255

                # import IPython; IPython.embed(); exit()
                if 'focal_length' in cam_opt_params:
                    renderer._set_focal_length(focal_length=new_focal_length[sidx])
                    # logger.debug(f'New focal length: {new_focal_length[sidx]}')
                    # logger.debug(f'New rotation mat: {new_camera_rotation}')
                else:
                    renderer._set_focal_length(focal_length=smplify_focal_length)

                smplify_pred = renderer.visualize_tb(
                    vertices=new_opt_vertices.detach()[sidx:sidx+1],
                    camera_translation=new_opt_cam_t.detach()[sidx:sidx+1],
                    images=images[sidx:sidx+1],
                    sideview=True,
                    camera_rotation=new_camera_rotation.detach()[sidx:sidx+1] if 'camera_rotation' in cam_opt_params else None,
                ).detach().cpu().numpy().transpose(1, 2, 0) * 255

                render_img = np.concatenate([hmr_pred, smplify_pred])
                render_img = np.clip(render_img, 0, 255).astype(np.uint8)

                if args.save:
                    os.makedirs(os.path.join(logdir, 'images'), exist_ok=True)
                    io.imsave(os.path.join(logdir, 'images', f'{batch_idx*curr_batch_size + sidx:06d}.png'), render_img)

                if args.show:
                    plt.figure(figsize=(10, 10))
                    if 'focal_length' in cam_opt_params:
                        plt.title(f'New f: {new_focal_length[sidx]}')
                    plt.imshow(render_img)
                    plt.show()
                    plt.close()

    logger.info('Final results')
    errors = np.array(errors)
    r_errors = np.array(r_errors)

    logger.info(f'MPJPE: {errors.mean():.4f}\u00B1{errors.std():.2f}, '
                f'PA-MPJPE: {r_errors.mean():.4f}\u00B1{r_errors.std():.2f}')

    if args.log:
        np.savez(
            os.path.join(logdir, 'results.npz'),
            pred_joints=pred_joints,
            pose=smpl_pose,
            betas=smpl_betas,
            camera=smpl_camera,
            mpjpe=errors,
            pa_mpjpe=r_errors,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path of the saved pre-trained model
    parser.add_argument('--dataset', type=str, default='mpi-inf-3dhp')
    parser.add_argument('--logdir', type=str, default='logs/smplify')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--cam_step_size', type=float, default=1e-2)
    parser.add_argument('--pose_step_size', type=float, default=1e-2)
    parser.add_argument('--cam_opt_iter', type=int, default=100)
    parser.add_argument('--pose_opt_iter', type=int, default=100)
    parser.add_argument('--focal_length', type=float, default=5000.)
    parser.add_argument('--cam_opt_params', default=['camera_translation', 'global_orient'], nargs='*')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--use_hmr', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--logs', default=[], nargs='*')

    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.logs[0], args.logs[1])
    else:
        main(args)

    # python scripts/smplify.py
    # --ckpt logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/tb_logs_pare/0_af6934cb2bdf49bc892a61c0306526a9/checkpoints/epoch=77.ckpt