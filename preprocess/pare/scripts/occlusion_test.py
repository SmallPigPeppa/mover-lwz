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
import matplotlib.pylab as plt
import skimage.io as io
import pytorch_lightning as pl
from skimage.transform import resize
from matplotlib import cm as mpl_cm, colors as mpl_colors
from scipy.stats import ttest_ind, ttest_1samp


from pare.core.single_image_trainer import SingleImageTrainer
from pare.utils.train_utils import load_pretrained_model
from pare.core.config import run_grid_search_experiments
from pare.utils.kp_utils import get_common_joint_names


def get_occluded_imgs(img, occ_size, occ_pixel, occ_stride):

    img_size = int(img.shape[-1])
    # Define number of occlusions in both dimensions
    output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))

    occ_img_list = []

    idx_dict = {}
    c = 0
    for h in range(output_height):
        for w in range(output_width):
            # Occluder window:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

            # Getting the image copy, applying the occluding window and classifying it:
            occ_image = img.clone()
            occ_image[:, :, h_start:h_end, w_start:w_end] = occ_pixel
            occ_img_list.append(occ_image)

            idx_dict[c] = (h,w)
            c += 1

    return torch.stack(occ_img_list, dim=0), idx_dict, output_height


def visualize_grid(image, heatmap, imgname=None, res_dict=None, save_dir=None):

    image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
    image = np.transpose(image.cpu().numpy(), (0, 2, 3, 1))[0]

    orig_heatmap = heatmap.copy()
    # heatmap = resize(heatmap, image.shape)
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_CUBIC)

    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap) # normalize between [0,1]

    title = ''
    if imgname:
        title += '/'.join(imgname.split('/')[-2:]) + '\n'

    if res_dict:
        title += f' err: {res_dict["mpjpe"]*1000:.2f}mm'
        title += f' r_err: {res_dict["pampjpe"]*1000:.2f}mm'

    w, h = 7, 2
    f, axarr = plt.subplots(h, w)
    f.set_size_inches((w*3, h*3))

    f.suptitle(title)
    joint_names = get_common_joint_names()

    for jid in range(len(joint_names)):
        axarr[jid // w, jid % w].axis('off')
        axarr[jid // w, jid % w].set_title(
            f'{joint_names[jid]} \n'
            f'min: {orig_heatmap[:,:,jid].min()*1000:.2f} '
            f'max: {orig_heatmap[:,:,jid].max()*1000:.2f}'
        )
        axarr[jid // w, jid % w].imshow(image)

        axarr[jid // w, jid % w].imshow(heatmap[:,:,jid], alpha=.5, cmap='jet', interpolation='none')

        # save heatmap as separate image
        if save_dir:
            cm = mpl_cm.get_cmap('jet')
            norm_gt = mpl_colors.Normalize()
            hm = cm(norm_gt(heatmap[:,:,jid]))[:, :, :3]
            hm = (hm*255).astype(np.uint8)
            io.imsave(os.path.join(save_dir, f'{jid:02d}.png'), hm)

            norm_hm = norm_gt(heatmap[:,:,jid])
            norm_hm = (norm_hm * 255).astype(np.uint8)
            io.imsave(os.path.join(save_dir, f'{jid:02d}_norm.png'), norm_hm)
            # nh = io.imread(os.path.join(save_dir, f'{jid:02d}_norm.png'))
            # np.save(os.path.join(save_dir, f'{jid:02d}_raw.npy'), heatmap[:,:,jid])
            # import IPython; IPython.embed(); exit()

    f.set_tight_layout(tight=True)


def analyze(take_mean=True, put_bars=False):
    from pare.utils.kp_utils import get_common_joint_names

    joint_names = get_common_joint_names()
    report_joints = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'head']
    joint_names[-2:] = ['head'] * 2

    # [idx for idx, x in enumerate(joint_names) if x.endswith(j)]

    # exps = [
    #     {
    #         'name': 'baseline',
    #         'dir': 'logs/pare/01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/01-09-2020_17-40-42_01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix_occlusion_test/output_images'
    #     },
    #     {
    #         'name': 'ours',
    #         'dir': 'logs/pare/03.09-pare_synth_occ_finetune/03-09-2020_15-46-30_03.09-pare_synth_occ_finetune_occlusion_test/output_images'
    #     },
    #     # {
    #     #     'name': 'synth_occ_pre',
    #     #     'dir': 'logs/pare/03.09-pare_synth_occ_pretrained/03-09-2020_15-41-09_03.09-pare_synth_occ_pretrained_occlusion_test/output_images'
    #     # }
    # ]

    # 'logs/pare/06.11-pare-ups_alldata_mpii3d-0.5/06-11-2020_10-41-45_06.11-pare-ups_alldata_mpii3d-0.5_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.5_0.3_0.2_train/occlusion_test_train'
    # 'logs/pare_coco/05.11-spin_ckpt_eval/logs/occlusion_test_train'


    exps = [
        {
            'name': 'SPIN',
            'dir': 'logs/pare_coco/05.11-spin_ckpt_eval/logs/occlusion_test_train/output_images'
        },
        {
            'name': 'HMR-EFT',
            'dir': 'logs/pare/27.10-spin_all_data/27-10-2020_21-05-03_27.10-spin_all_data_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train/occlusion_test_train/output_images'
        },
        {
            'name': 'PARE',
            'dir': 'logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train/occlusion_test_train/output_images'
        },
        # {
        #     'name': 'synth_occ_pre',
        #     'dir': 'logs/pare/03.09-pare_synth_occ_pretrained/03-09-2020_15-41-09_03.09-pare_synth_occ_pretrained_occlusion_test/output_images'
        # }
    ]

    results = []

    t_test_arr = []
    for exp in exps:
        exp_name = exp['name']
        exp_dir = exp['dir']
        res_files = sorted([os.path.join(exp_dir, x) for x in os.listdir(exp_dir) if x.endswith('.pkl')])[:243]

        per_joint_max_err = []
        for res_f in tqdm(res_files):
            res_dict = joblib.load(res_f)
            if take_mean:
                per_j_mpjpe = res_dict['mpjpe_heatmap'].mean(0).mean(0)
            else:
                per_j_mpjpe = res_dict['mpjpe_heatmap'].max(0).max(0)

            cum_err = []
            for j in report_joints:
                e = per_j_mpjpe[[idx for idx, x in enumerate(joint_names) if x.endswith(j)]].mean()
                cum_err.append(e)

            per_joint_max_err.append(np.array(cum_err))

        per_joint_max_err = np.array(per_joint_max_err)

        t_test_arr.append(per_joint_max_err.mean(-1))
        mean, std = per_joint_max_err.mean(0) * 1000, per_joint_max_err.std(0) * 1000
        results.append([exp_name, mean, std])

    # tt = ttest_ind(t_test_arr[0], t_test_arr[1])
    # tt = ttest_1samp(t_test_arr[0])
    # breakpoint()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18


    x = np.arange(len(report_joints))  # the label locations
    width = 0.3  # the width of the bars
    pos = [-width, 0, width]
    # pos = [-width/2, width/2]
    # ylim = [56, 61]
    fig_size = (8, 4)

    fig, ax = plt.subplots()

    plt.grid(linestyle='dashed')
    for idx, res in enumerate(results):
        # data = df.loc[(df['mode'] == model_type) & (df['drop'] == drop)].groupby(['seql'])['result'].mean()
        # data = [data.to_dict()[x] for x in seqlen]
        #
        # std = df.loc[(df['mode'] == model_type) & (df['drop'] == drop)].groupby(['seql'])['result'].std()
        # std = [std.to_dict()[x] for x in seqlen]
        exp_name = res[0]
        data = res[1]
        std = res[2]
        if put_bars:
            rects = ax.bar(x + pos[idx], data, width,
                           yerr=std,
                           label=f'{exp_name}',
                           error_kw=dict(lw=1, capsize=3, capthick=1, ecolor='gray'),
                           )
        else:

            rects = ax.bar(x + pos[idx], data, width,
                           # yerr=std,
                           label=f'{exp_name}',
                           # error_kw=dict(lw=1, capsize=3, capthick=1, ecolor='gray'),
                           )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MPJPE (mm)')
    # ax.set_xlabel('Joints')
    # ax.set_ylim(*ylim)
    ax.set_title(f'Occlusion Sensitivity Per Joint')
    ax.set_ylabel('Average Joint Error (mm)')
    ax.set_xticks(x)
    ax.set_xticklabels([x.capitalize() for x in report_joints])
    ax.legend()

    fig.tight_layout()
    fig.set_size_inches(*fig_size)
    if take_mean:
        if put_bars:
            plt.savefig(f'logs/figures/occlusion_per_joint_analysis_bar_mean.pdf')
        else:
            plt.savefig(f'logs/figures/occlusion_per_joint_analysis_mean.pdf')
    else:
        if put_bars:
            plt.savefig(f'logs/figures/occlusion_per_joint_analysis_bar.pdf')
        else:
            plt.savefig(f'logs/figures/occlusion_per_joint_analysis.pdf')
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--ckpt', type=str, default=None)  # Path of the saved pre-trained model
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default='100')  # save frequency
    parser.add_argument('--dataset', type=str, default='3dpw_3doh')  # Path of the input image
    parser.add_argument('--occ_size', type=int, default='40')  # Size of occluding window
    parser.add_argument('--pixel', type=int, default='0')  # Occluding window - pixel values
    parser.add_argument('--stride', type=int, default='40')  # Occlusion Stride

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=0,
        cfg_file=args.cfg,
        bid=300,
        use_cluster=False,
        memory=16000,
        script='occlusion_test.py',
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = SingleImageTrainer(hparams=hparams).to(device)
    model = model.eval()

    val_images_errors = []

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    dataloader = model.val_dataloader()[0]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % args.save_freq != 0:
            continue

        logger.info(f'Processing {batch_idx} / {len(dataloader)} "{batch["imgname"]}"')

        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(device)

        occluded_images, idx_dict, output_size = get_occluded_imgs(
            batch['img'],
            occ_size=args.occ_size,
            occ_pixel=args.pixel,
            occ_stride=args.stride
        )
        ratio = hparams.DATASET.RENDER_RES / hparams.DATASET.IMG_RES
        occluded_images_disp, idx_dict, output_size = get_occluded_imgs(
            batch['disp_img'],
            occ_size=int(round(args.occ_size * ratio)),
            occ_pixel=args.pixel,
            occ_stride=int(round(args.stride * ratio)),
        )

        mpjpe_heatmap = np.zeros((output_size, output_size, 14))
        pampjpe_heatmap = np.zeros((output_size, output_size, 14))

        orig_image = batch['disp_img']

        model.hparams.TESTING.SAVE_MESHES = True
        orig_res_dict = model.validation_step(batch, batch_idx, dataloader_nb=0, vis=True, save=True)

        val_images_errors.append([orig_res_dict['mpjpe'], orig_res_dict['pampjpe']])

        save_dir = os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}_occlusion')
        os.makedirs(save_dir, exist_ok=True)

        mesh_save_dir = os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}_meshes')
        os.makedirs(mesh_save_dir, exist_ok=True)

        for occ_img_idx in tqdm(range(occluded_images.shape[0])):
            batch['img'] = occluded_images[occ_img_idx]
            batch['disp_img'] = occluded_images_disp[occ_img_idx]

            model.hparams.TESTING.SAVE_MESHES = True
            result_dict = model.validation_step(
                batch, occ_img_idx, dataloader_nb=0, vis=True, save=False,
                mesh_save_dir=mesh_save_dir,
            )

            cv2.imwrite(
                os.path.join(save_dir, f'result_{occ_img_idx:05d}.jpg'),
                result_dict['vis_image']
            )

            mpjpe_heatmap[idx_dict[occ_img_idx]] = result_dict['per_mpjpe'][0]
            pampjpe_heatmap[idx_dict[occ_img_idx]] = result_dict['per_pampjpe'][0]

        # heatmap_size = int(math.sqrt(occluded_images.shape[0]))
        # mpjpe_heatmap = np.array(mpjpe_heatmap).reshape((heatmap_size, heatmap_size, -1)).T
        # pampjpe_heatmap = np.array(pampjpe_heatmap).reshape((heatmap_size, heatmap_size, -1)).T

        command = [
            'ffmpeg', '-y',
            '-framerate', '15',
            '-i', f'{save_dir}/result_%05d.jpg',
            '-c:v', 'libx264', '-profile:v', 'high',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}.mp4')
        ]
        logger.info(f'Running {"".join(command)}')
        subprocess.call(command)
        # shutil.rmtree(save_dir)

        save_dir = save_dir.replace('_occlusion', '_hm')
        os.makedirs(save_dir, exist_ok=True)
        fig = plt.figure()
        visualize_grid(orig_image, mpjpe_heatmap, res_dict=orig_res_dict, save_dir=save_dir)
        plt.savefig(os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}_mpjpe_hm.png'))
        # plt.show()

        visualize_grid(orig_image, pampjpe_heatmap, res_dict=orig_res_dict)
        plt.savefig(os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}_pampjpe_hm.png'))
        # plt.show()

        plt.close(fig)
        # import IPython; IPython.embed(); exit()

        orig_res_dict['mpjpe_heatmap'] = mpjpe_heatmap
        orig_res_dict['pampjpe_heatmap'] = pampjpe_heatmap
        orig_res_dict['imgname'] = batch['imgname']

        del orig_res_dict['vis_image']

        joblib.dump(
            value=orig_res_dict,
            filename=os.path.join(hparams.LOG_DIR, 'output_images', f'result_00_{batch_idx:05d}.pkl'),
        )

        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close('all')

        # break

    save_path = os.path.join(hparams.LOG_DIR, 'val_images_error.npy')
    logger.info(f'Saving the errors of images {save_path}')
    np.save(save_path, np.asarray(val_images_errors))

if __name__ == '__main__':
    # main()
    analyze(take_mean=False)

# python scripts/occlusion_test.py --cfg configs/pare_cfg.yaml --ckpt logs/pare/01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/01-09-2020_17-40-42_01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/tb_logs_pare/0_c2083df7f5964bfa881e0b5097c76639/checkpoints/epoch\=8.ckpt