import os
import torch
import argparse
import numpy as np
import skimage.io as io
from loguru import logger
from torch.utils.data import DataLoader

from pare.models import SMPL
from pare.utils.renderer import Renderer
# from pare.models.hmr_adf_dropout import HMR_ADF_DROPOUT
from pare.models import HMR
from pare.utils.train_utils import load_pretrained_model
from pare.dataset import BaseDataset
from pare.core.config import get_hparams_defaults, SMPL_MODEL_DIR
from pare.utils.vis_utils import color_vertices


def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if training:
                m.train()
            else:
                m.eval()

    return net


def main(args):
    cfg = get_hparams_defaults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = HMR_ADF_DROPOUT().to(device)
    # model = HMR().to(device)
    logger.info(f'Dropout rate for uncertainty estimation: {args.dropout}')
    model = HMR(p=args.dropout).to(device)
    model.eval()

    logger.info(f'Loading pretrained model from {args.ckpt}')
    ckpt = torch.load(args.ckpt)['state_dict']
    load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )

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
        num_workers=8,
        batch_size=1,
    )

    uncertainty_values = []

    os.makedirs(f'logs/uncertainty_logs_spin/results_{args.dropout}', exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % args.save_freq != 0:
            continue

        logger.info(f'Processing {batch_idx} / {len(dataloader)} "{batch["imgname"]}"')

        images = batch['img'].to(device)

        # get the original prediction
        model = set_training_mode_for_dropout(model, training=False)
        orig_output = model(images)

        # compute model uncertainty
        model = set_training_mode_for_dropout(model, training=True)
        outputs = [model(images) for i in range(args.num_samples)]

        pred_pose = torch.stack([out['pred_pose'] for out in outputs])
        pred_shape = torch.stack([out['pred_shape'] for out in outputs])
        pred_cam = torch.stack([out['pred_cam'] for out in outputs])

        pred_pose_var = pred_pose.mean(-1).mean(-1).var(0)
        pred_shape_var = pred_shape.var(0)
        pred_cam_var = pred_cam.var(0)

        orig_output = outputs[5]

        # import IPython; IPython.embed(); exit()
        per_joint_label = pred_pose_var.detach().cpu().numpy()
        ##### RENDER #####

        uncertainty_values += list(per_joint_label)
        uv = np.array(uncertainty_values)
        logger.info(f'Running pose uncertainty statistics: '
                    f'mean: {uv.mean()}, min: {uv.min()}, max: {uv.max()}')

        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        render_images = renderer.visualize_tb(
            vertices=orig_output['smpl_vertices'].detach(),
            camera_translation=orig_output['pred_cam_t'].detach(),
            images=images,
            # kp_2d=orig_output['smpl_joints2d'].detach(),
            # skeleton_type='spin',
            sideview=True,
            vertex_colors=np.expand_dims(color_vertices(per_joint_label, alpha=0.8), 0)
        )
        # breakpoint()
        render_images = render_images.detach().cpu().numpy().transpose(1, 2, 0) * 255
        render_images = np.clip(render_images, 0, 255).astype(np.uint8)
        io.imsave(f'logs/uncertainty_logs_spin/results_{args.dropout}/{batch_idx:05d}.png', render_images)
        # plt.figure(figsize=(20, 6))
        # plt.imshow(render_images)
        # plt.savefig(f'logs/uncertainty/{batch_idx:04d}.png')
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, required=True)  # Path of the saved pre-trained model
    parser.add_argument('--dataset', type=str, default='3dpw')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--save_freq', type=int, default=100)

    args = parser.parse_args()
    main(args)

# python scripts/uncertainty_estimation.py --ckpt logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/tb_logs_pare/0_af6934cb2bdf49bc892a61c0306526a9/checkpoints/epoch\=77.ckpt