import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from pare.utils.mmpose import run_mmpose
from pare.eft.eft_fitter import EFTFitter
from pare.utils.kp_utils import convert_kps
from pare.dataset.inference import Inference
from pare.eft.config import get_cfg_defaults


def main(args):
    image_folder = args.image_folder

    # Joints2d should be of size [num_images, num_keypoints, 3]
    # Last 3 dimension is -> x, y, confidence
    # For ground truth keypoints confidence becomes binary visibility label
    # Here we use mmpose for demo purposes
    logger.info('Running MMPOSE on input images')
    joints2d = run_mmpose(image_folder, show_results=False)

    # Prepare EFT config for running on in the wild images
    hparams = get_cfg_defaults()
    hparams.LOG = True
    hparams.LOG_DIR = args.log_dir
    hparams.DATASET.RENDER_RES = hparams.DATASET.IMG_RES
    hparams.LOSS.OPENPOSE_TRAIN_WEIGHT = 1.0
    hparams.DATASET.VAL_DS = 'custom'

    if args.ckpt != '' and os.path.isfile(args.ckpt):
        hparams.PRETRAINED_CKPT = args.ckpt

    device = 'cuda' if torch.cuda.is_available() else sys.exit('CUDA should be available')

    eft_fitter = EFTFitter(hparams=hparams)

    # Convert from source keypoint format to SPIN keypoints
    # Check `pare/utils/kp_utils.py` to see the source keypoint format is existent
    # if not create a function called `get_<name_of_dataset>_joint_names()` in `pare/utils/kp_utils.py`
    # that returns the name of the joints
    # then `convert_kps` function will handle the rest
    joints2d = convert_kps(joints2d, src='mmpose', dst='spin')

    dataset = Inference(
        image_folder=image_folder,
        bboxes=None,
        joints2d=joints2d,
        scale=1.0,
        return_dict=True,
        crop_size=hparams.DATASET.IMG_RES,
        normalize_kp2d=True,
    )

    dataloader = DataLoader(dataset, batch_size=hparams.DATASET.BATCH_SIZE, num_workers=8)

    output_cam, output_verts, output_pose, output_betas, output_joints3d = [], [], [], [], []

    for batch_nb, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = eft_fitter.finetune_step(batch, batch_nb)

        batch['disp_img'] = batch['img']
        # This saves the EFT renderings to hparams.LOG_DIR directory
        eft_fitter.visualize_results(batch, output, batch_nb, has_error_metrics=False)

        output_cam.append(output['pred_cam'])
        output_verts.append(output['smpl_vertices'])
        output_pose.append(output['pred_pose'])
        output_betas.append(output['pred_shape'])
        output_joints3d.append(output['smpl_joints3d'])

    output_cam = torch.cat(output_cam, dim=0)
    output_verts = torch.cat(output_verts, dim=0)
    output_pose = torch.cat(output_pose, dim=0)
    output_betas = torch.cat(output_betas, dim=0)
    output_joints3d = torch.cat(output_joints3d, dim=0)

    with torch.no_grad():
        output_cam = output_cam.cpu().numpy()
        output_verts = output_verts.cpu().numpy()
        output_pose = output_pose.cpu().numpy()
        output_betas = output_betas.cpu().numpy()
        output_joints3d = output_joints3d.cpu().numpy()

    np.savez(
        os.path.join(hparams.LOG_DIR, f'eft_output.npz'),
        output_cam=output_cam,
        output_verts=output_verts,
        output_pose=output_pose,
        output_betas=output_betas,
        output_joints3d=output_joints3d,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True, help='input image folder')
    parser.add_argument('--log_dir', type=str, required=True, help='output folder folder to save results')
    parser.add_argument('--ckpt', type=str, default='', required=False, help='SPIN pretrained checkpoint')

    args = parser.parse_args()

    main(args)