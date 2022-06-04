import os
import cv2
import math
import time
import torch
import shutil
import joblib
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from loguru import logger
import skimage.io as io
import matplotlib.pylab as plt
import pytorch_lightning as pl
from skimage.transform import resize
from matplotlib import cm as mpl_cm, colors as mpl_colors

from pare.core.single_image_trainer import SingleImageTrainer
from pare.utils.train_utils import load_pretrained_model
from pare.core.config import run_grid_search_experiments
from pare.utils.kp_utils import get_common_joint_names


def visualize_grid(image, heatmap, imgname=None, res_dict=None, save_dir=None):

    image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
    image = np.transpose(image.cpu().numpy(), (0, 2, 3, 1))[0]

    orig_heatmap = heatmap.copy()
    # heatmap = resize(heatmap, image.shape)
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_CUBIC)

    # heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap) # normalize between [0,1]

    title = ''
    if imgname:
        title += '/'.join(imgname.split('/')[-2:]) + '\n'

    if res_dict:
        title += f' err: {res_dict["mpjpe"]*1000:.2f}mm'
        title += f' r_err: {res_dict["pampjpe"]*1000:.2f}mm'

    w = h = math.ceil(math.sqrt(heatmap.shape[-1]))

    f, axarr = plt.subplots(h, w)
    f.set_size_inches((w, h))

    f.suptitle(title)
    # joint_names = get_common_joint_names()

    for act_id in range(heatmap.shape[-1]):
        axarr[act_id // w, act_id % w].axis('off')
        # axarr[act_id // w, act_id % w].set_title(
        #     f'{act_id} '
        #     f'min: {orig_heatmap[:,:,act_id].min():.2f} '
        #     f'max: {orig_heatmap[:,:,act_id].max():.2f}'
        # )

        axarr[act_id // w, act_id % w].imshow(image)
        axarr[act_id // w, act_id % w].imshow(heatmap[:,:,act_id], alpha=.5, cmap='jet', interpolation='none')

        # save heatmap as separate image
        if save_dir:
            cm = mpl_cm.get_cmap('jet')
            norm_gt = mpl_colors.Normalize()
            hm = cm(norm_gt(heatmap[:, :, act_id]))[:, :, :3]
            io.imsave(os.path.join(save_dir, f'{act_id:02d}.png'), hm)

    f.set_tight_layout(tight=True)


def visualize_activations(model):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.model.head.keypoint_deconv_layers.register_forward_hook(get_activation('kp_conv'))

    act = activation['kp_conv'].squeeze()


def main(hparams, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = SingleImageTrainer(hparams=hparams).to(device)
    model = model.eval()

    save_dir = os.path.join(hparams.LOG_DIR, 'output_images')
    os.makedirs(save_dir, exist_ok=True)

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    dataloader = model.val_dataloader()[0]

    model.hparams.TESTING.SAVE_MESHES = True

    for batch_idx, batch in enumerate(dataloader):

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        if batch_idx % args.save_freq != 0:
            continue

        logger.info(f'Processing {batch_idx} / {len(dataloader)} "{batch["imgname"]}"')

        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(device)

        orig_image = batch['disp_img']

        model.model.head.keypoint_deconv_layers.register_forward_hook(get_activation('kp_conv'))

        orig_res_dict = model.validation_step(batch, batch_idx, dataloader_nb=0, vis=True, save=True)

        activation = activation['kp_conv'].squeeze()
        activation = activation.detach().cpu().numpy().transpose(1, 2, 0)

        most_activated = activation[:, :, activation.sum(axis=0).sum(axis=0).argsort()[-64:]]
        # import IPython; IPython.embed()
        save_dir = os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}')
        os.makedirs(save_dir, exist_ok=True)
        fig = plt.figure()
        visualize_grid(orig_image, most_activated, res_dict=orig_res_dict, save_dir=save_dir)
        plt.savefig(os.path.join(hparams.LOG_DIR, 'output_images', f'result_{batch_idx:05d}_activations.png'))
        # plt.show()

        plt.close(fig)
        plt.close('all')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--ckpt', type=str, default=None) # Path of the saved pre-trained model
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default='100')  # Occlusion Stride
    parser.add_argument('--dataset', type=str, default='3dpw')  # Path of the input image

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=0,
        cfg_file=args.cfg,
        bid=300,
        use_cluster=False,
        memory=16000,
        script='visualize_activations.py',
    )

    if args.ckpt is not None:
        logger.info(f'Pretrained checkpoint is \"{args.ckpt}\"')
        hparams.TRAINING.PRETRAINED_LIT = args.ckpt

    if args.dataset is not None:
        logger.info(f'Test dataset is \"{args.dataset}\"')
        hparams.DATASET.VAL_DS = args.dataset

    if args.batch_size is not None:
        logger.info(f'Testing batch size \"{args.batch_size}\"')
        hparams.DATASET.BATCH_SIZE = args.batch_size

    hparams.RUN_TEST = True
    # from threadpoolctl import threadpool_limits
    # with threadpool_limits(limits=1):
    main(hparams, args)

# python scripts/occlusion_test.py --cfg configs/pare_cfg.yaml --ckpt logs/pare/01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/01-09-2020_17-40-42_01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/tb_logs_pare/0_c2083df7f5964bfa881e0b5097c76639/checkpoints/epoch\=8.ckpt