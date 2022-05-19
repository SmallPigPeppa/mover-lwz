import os
import cv2
import sys
import torch
import joblib
import shutil
import argparse
import itertools
import subprocess
import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib as mpl
from os.path import join
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('.')

from sklearn.cluster import KMeans

plt.rcParams["font.family"] = "Georgia"
# plt.rcParams["font.size"] = 20

from pare.utils.vis_utils import overlay_heatmaps
from pare.utils.kp_utils import get_common_joint_names, get_common_paper_joint_names, get_smpl_paper_joint_names
# from .occlusion_vis_mesh import normalize_heatmaps

PARE_DIR = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train'
SPIN_DIR = 'logs/pare_coco/05.11-spin_ckpt_eval'
HMR_DIR = 'logs/pare/27.10-spin_all_data/27-10-2020_21-05-03_27.10-spin_all_data_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train'
PARE_ATTENTION_DIR = 'logs/pare/06.11-pare-ups_alldata_mpii3d-0.5/06-11-2020_10-41-59_06.11-pare-ups_alldata_mpii3d-0.5_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train'

ALPHA = 0.6
FIG_EXT = '.pdf'

def quantize_img(img, n_colors=8):
    arr = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    img = centers[labels].reshape(img.shape).astype('uint8')
    return img


def occlusion_sensitivity_hm(debug=True):
    images_f = [
        'result_00_02400.jpg',
        'result_00_01200.jpg',
        'result_00_02700.jpg',
        'result_00_35100.jpg',
        'result_00_33100.jpg',
        'result_00_26600.jpg',
        'result_00_27500.jpg',
        'result_00_27400.jpg',
        'result_00_27200.jpg',
        'result_00_26600.jpg',
        'result_00_01100.jpg',
        'result_00_01200.jpg',
        'result_00_03800.jpg',
        'result_00_04200.jpg',
        'result_00_04600.jpg',
        'result_00_05100.jpg',
        'result_00_35500.jpg',
        'result_00_34600.jpg',
        'result_00_33900.jpg',
        'result_00_27900.jpg',
        'result_00_27400.jpg',
        'result_00_27300.jpg',
        'result_00_26900.jpg',
        'result_00_26800.jpg',
        'result_00_22200.jpg',
        'result_00_21900.jpg',
        'result_00_20900.jpg',
        'result_00_16300.jpg',
        'result_00_15000.jpg',
        'result_00_13300.jpg',
        'result_00_13200.jpg',
        'result_00_13100.jpg',
        'result_00_12400.jpg',
        'result_00_09900.jpg',
        'result_00_09700.jpg',
        'result_00_09200.jpg',
    ]

    images_f = [join(SPIN_DIR, 'logs/occlusion_test_train', f'output_images/{x}') for x in images_f]

    joint_names = get_common_paper_joint_names()


    for im_f in images_f:
        print(os.path.basename(im_f))
        image = io.imread(im_f.replace('output_images', 'output_meshes'))
        obj_f = im_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj').replace(PARE_ATTENTION_DIR,
                                                                                               PARE_DIR)
        overlay_f = obj_f.replace('.obj', '_overlay.png')
        if not os.path.isfile(overlay_f):
            run_blender_on_single_obj(
                obj_f=obj_f, color='turkuaz', im_size=image.shape[0],
            )

        img_blender = io.imread(overlay_f)

        pos = []

        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(30, 8),
                               gridspec_kw={'wspace': 0.005, 'hspace': 0.2})
        # fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        for i in range(2):
            for j in range(8):
                ax[i, j].axis('off')

                if not [i, j] in [[0, 0], [1, 0]]:
                    pos.append((i, j))

        ax[0, 0].imshow(image)
        ax[1, 0].imshow(img_blender)

        for idx, jn in enumerate(joint_names):
            face_err_path = os.path.join(im_f.replace('.jpg', '_hm'), f'{idx:02d}_face_errors.npy')
            face_err = np.load(face_err_path)
            face_err = face_err[face_err > 0.0] * 1000.

            hm_path = os.path.join(im_f.replace('.jpg', '_hm'), f'{idx:02d}_norm.png')
            hm = io.imread(hm_path)

            # ohm = overlay_heatmaps(image.copy(), hm=hm, alpha=0.5)
            # print(pos[idx])
            ax[pos[idx]].set_title(f'{jn}\nmin: {face_err.min():.1f} - max: {face_err.max():.1f}')
            ax[pos[idx]].imshow(image)
            ax[pos[idx]].imshow(hm, cmap='turbo', alpha=ALPHA)

        save_f = join('logs/figures/occlusion_heatmaps',
                      f'{os.path.basename(im_f).replace(".jpg", f"_occlusion_hm{FIG_EXT}")}')
        fig.savefig(save_f, bbox_inches='tight', transparent=True)
        plt.close('all')
        # plt.show()

@plt.style.context('dark_background')
def attention_fig(debug=True, quantize=False):
    # result_dir = join(PARE_DIR, 'vis_parts', 'output_images')
    #
    # images_f = sorted([os.path.join(result_dir, x) for x in os.listdir(result_dir) if x.endswith('.jpg')])
    #
    # for imf in images_f:
    #     print(imf)
    #     cv2.imshow('attention_img', cv2.imread(imf))
    #     cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    images_f = [
        'result_00_05216.jpg',
        'result_00_05120.jpg',
        'result_00_10720.jpg',
        'result_00_03520.jpg',
        'result_00_03424.jpg',
        'result_00_08288.jpg',
        'result_00_08384.jpg',
        'result_00_10624.jpg',
    ]
    images_f = [join(PARE_ATTENTION_DIR, f'vis_parts/output_images/{x}') for x in images_f]
    # images_f = [join(PARE_ATTENTION_DIR, 'vis_parts/output_images/result_00_05216.jpg')]
    # images_f = [join(PARE_ATTENTION_DIR, 'vis_parts/output_images/result_00_00064.jpg')]

    joints_to_visualize = [0, 4, 5, 7, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    for im_f in images_f:
        image = io.imread(im_f.replace('output_images', 'output_meshes'))

        obj_f = im_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj').replace(PARE_ATTENTION_DIR, PARE_DIR)
        overlay_f = obj_f.replace('.obj', '_overlay.png')
        if not os.path.isfile(overlay_f):
            run_blender_on_single_obj(
                obj_f=obj_f, color='turkuaz', im_size=image.shape[0],
            )

        img_blender = io.imread(overlay_f)

        pos = []

        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(32,8),
                               gridspec_kw={'wspace': 0.01, 'hspace': 0.15})
        fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        for i in range(2):
            for j in range(8):
                ax[i,j].axis('off')

                if not [i,j] in [[0,0],[1,0]]:
                    pos.append((i,j))
        ax[0,0].set_title('(a) Input Image')
        ax[0,0].imshow(image)
        ax[1,0].set_title('(b) PARE result')
        ax[1,0].imshow(img_blender)

        # pos = [
        #     (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        #     (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        # ]

        for idx, i in enumerate(joints_to_visualize):
            i += 1
            jn = (['background'] + get_smpl_paper_joint_names())[i]
            hm_path = os.path.join(im_f.replace('.jpg', ''), f'{i:02d}.png')
            hm = io.imread(hm_path)

            if quantize:
                hm = quantize_img(hm, n_colors=6)

            ohm = overlay_heatmaps(image.copy(), hm=hm, alpha=0.5)
            print(pos[idx])
            ax[pos[idx]].set_title(jn)
            ax[pos[idx]].imshow(ohm)
        FIG_EXT = '.png'
        plt.close('all')
        if quantize:
            save_f = join('logs/figures/attention', f'{os.path.basename(im_f).replace(".jpg", f"_quant{FIG_EXT}")}')
        else:
            save_f = join('logs/figures/attention', f'{os.path.basename(im_f).replace(".jpg", FIG_EXT)}')

        fig.savefig(save_f, bbox_inches='tight', transparent=True)
        plt.show()


def normalize_heatmaps(method_1_dir, method_2_dir, img, joint_id, im_size=480):
    j = joint_id # 13

    min_ = np.inf
    max_ = -np.inf

    hm_arr = []
    for m_idx, m in enumerate((method_1_dir, method_2_dir)):
        pkl = os.path.join(m, 'output_images', img.replace('.jpg', '.pkl'))

        hm = joblib.load(pkl)['mpjpe_heatmap'].mean(-1)
        hm = cv2.copyMakeBorder(hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
        hm = cv2.resize(hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

        hm_arr.append(hm)

    minmax = (min_, max_)

    heatmap = np.stack(hm_arr)

    # heatmap = hist_eq(heatmap)
    heatmap_orig = heatmap.copy()
    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)  # normalize between [0,1]

    cm = mpl.cm.get_cmap('jet')

    heatmap_color = []
    for i, h in enumerate(heatmap):
        # print(h.min(), h.max())
        h_c = cm(h)[:, :, :3]
        h_c = (h_c * 255).astype(np.uint8)
        heatmap_color.append(h_c)

    heatmap_orig *= 1000.

    return heatmap_color, heatmap_orig


def occlusion_meshes(pare=False):
    img_dir = 'logs/figures/occlusion_meshes'
    method = ['pare', 'spin']

    # joint_names = get_common_joint_names()

    joint_names = [
        "all",
        "rankle",  # 0  "lankle",    # 0
        "rknee",  # 1  "lknee",     # 1
        "rhip",  # 2  "lhip",      # 2
        "lhip",  # 3  "rhip",      # 3
        "lknee",  # 4  "rknee",     # 4
        "lankle",  # 5  "rankle",    # 5
        "rwrist",  # 6  "lwrist",    # 6
        "relbow",  # 7  "lelbow",    # 7
        "rshoulder",  # 8  "lshoulder", # 8
        "lshoulder",  # 9  "rshoulder", # 9
        "lelbow",  # 10  "relbow",    # 10
        "lwrist",  # 11  "rwrist",    # 11
        "neck",  # 12  "neck",      # 12
        "headtop",  # 13  "headtop",   # 13
    ]
    if pare:
        for jn in joint_names:
            print(jn)
            spin_f = io.imread(f'{img_dir}/spin_{jn}_front.png')
            spin_b = io.imread(f'{img_dir}/spin_{jn}_back.png')
            colorb = io.imread(f'{img_dir}/pare_{jn}_colorbar.png')
            pare_f = io.imread(f'{img_dir}/pare_{jn}_front.png')
            pare_b = io.imread(f'{img_dir}/pare_{jn}_back.png')

            fig, ax = plt.subplots(
                ncols=5,
                figsize=(32, 12),
                gridspec_kw={
                    'width_ratios': [20,20,2,20,20],
                    'height_ratios': [1,],
                }
            )
            fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

            for i in range(5):
                ax[i].axis('off')

            # if set_title:
            #     ax[0].set_title('Input Image')
            #     ax[1].set_title('SPIN')
            #     ax[2].set_title('PARE')

            ax[0].imshow(spin_f)
            ax[1].imshow(spin_b)
            ax[2].imshow(colorb)
            ax[3].imshow(pare_f)
            ax[4].imshow(pare_b)

            # plt.show()
            plt.close('all')
            save_f = join(img_dir, f'paper_fig_{jn}.png')
            fig.savefig(save_f, bbox_inches='tight', transparent=True)

    for jn in joint_names:
        print(jn)
        spin_f = io.imread(f'{img_dir}/spin_{jn}_front.png')
        spin_b = io.imread(f'{img_dir}/spin_{jn}_back.png')
        colorb = io.imread(f'{img_dir}/pare_{jn}_colorbar.png')


        fig, ax = plt.subplots(
            ncols=3,
            figsize=(19.2, 12),
            gridspec_kw={
                'width_ratios': [20,20,2],
                'height_ratios': [1,],
            }
        )
        fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

        for i in range(3):
            ax[i].axis('off')

        # if set_title:
        #     ax[0].set_title('Input Image')
        #     ax[1].set_title('SPIN')
        #     ax[2].set_title('PARE')

        ax[0].imshow(spin_f)
        ax[1].imshow(spin_b)
        ax[2].imshow(colorb)
        # ax[3].imshow(pare_f)
        # ax[4].imshow(pare_b)

        # plt.show()
        plt.close('all')
        save_f = join(img_dir, f'paper_fig_spin_{jn}.png')
        fig.savefig(save_f, bbox_inches='tight', transparent=True)


def qual_coco(image_f, im_size=480, set_title=True, output_folder='coco_v2', render_rot=False):
    plt.rcParams["font.size"] = 20

    print(image_f)
    pare_f = image_f # join(PARE_DIR, image_f)
    spin_f = join(SPIN_DIR, image_f)
    hmr_f = join(HMR_DIR, image_f)

    input_image = pare_f.replace('output_images', 'output_meshes')

    pare_render = input_image.replace('.jpg', '_overlay.png')
    spin_render = pare_render.replace(PARE_DIR, SPIN_DIR)
    hmr_render = pare_render.replace(PARE_DIR, HMR_DIR)

    # if not os.path.isfile(pare_render):
    run_blender_on_single_obj(
        obj_f=pare_render.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size,
    )

    # if not os.path.isfile(spin_render):
    run_blender_on_single_obj(
        obj_f=spin_render.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size,
    )

    # if not os.path.isfile(hmr_render):
    run_blender_on_single_obj(
        obj_f=hmr_render.replace('_overlay.png', '.obj'), color='light_red', im_size=im_size,
    )

    os.makedirs(f'logs/figures/{output_folder}_input_images', exist_ok=True)
    print(f'copying {input_image}')
    shutil.copy2(input_image, f'logs/figures/{output_folder}_input_images')

    input_image = io.imread(input_image)
    spin_result = io.imread(spin_render)
    pare_result = io.imread(pare_render)
    hmr_result = io.imread(hmr_render)

    fig, ax = plt.subplots(
        ncols=4,
        figsize=(24, 9)
    )

    for i in range(4):
        ax[i].axis('off')

    if set_title:
        ax[0].set_title('Input Image')
        ax[1].set_title('SPIN')
        ax[2].set_title('HMR-EFT')
        ax[3].set_title('PARE')

    ax[0].imshow(input_image)
    ax[1].imshow(spin_result)
    ax[2].imshow(hmr_result)
    ax[3].imshow(pare_result)
    fig.set_tight_layout(True)

    save_dir = f'logs/figures/{output_folder}'
    os.makedirs(save_dir, exist_ok=True)
    fig.set_tight_layout(True)
    save_f = join(save_dir, os.path.basename(image_f))
    print(f'Saving \"{save_f}\"')
    fig.savefig(save_f.replace('.jpg', '.png'), bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close('all')


def qual_coco_fail(image_f, im_size=480, set_title=True, output_folder='coco_fail', render_rot=False):
    plt.rcParams["font.size"] = 20

    print(image_f)
    pare_f = image_f # join(PARE_DIR, image_f)

    input_image = pare_f.replace('output_images', 'output_meshes')

    pare_render = input_image.replace('.jpg', '_overlay.png')
    pare_render_side = input_image.replace('.jpg', '_270_rot_render.png')

    if not os.path.isfile(pare_render):
        run_blender_on_single_obj(
            obj_f=pare_render.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size,
        )
    if not os.path.isfile(pare_render_side):
        run_blender_on_single_obj(
            obj_f=pare_render.replace('_overlay.png', '_270_rot.obj'), color='turkuaz', im_size=im_size,
        )

    # if not os.path.isfile(spin_render):
    # run_blender_on_single_obj(
    #     obj_f=spin_render.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size,
    # )

    # if not os.path.isfile(hmr_render):
    # run_blender_on_single_obj(
    #     obj_f=hmr_render.replace('_overlay.png', '.obj'), color='light_red', im_size=im_size,
    # )

    os.makedirs(f'logs/figures/{output_folder}_input_images', exist_ok=True)
    print(f'copying {input_image}')
    shutil.copy2(input_image, f'logs/figures/{output_folder}_input_images')

    input_image = io.imread(input_image)
    pare_result = io.imread(pare_render)
    pare_result_side = io.imread(pare_render_side)

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(24, 9)
    )

    for i in range(3):
        ax[i].axis('off')

    if set_title:
        ax[0].set_title('Input Image')
        ax[1].set_title('SPIN')
        ax[2].set_title('HMR-EFT')
        ax[3].set_title('PARE')

    ax[0].imshow(input_image)
    ax[1].imshow(pare_result)
    ax[2].imshow(pare_result_side)
    fig.set_tight_layout(True)

    save_dir = f'logs/figures/{output_folder}'
    os.makedirs(save_dir, exist_ok=True)
    fig.set_tight_layout(True)
    save_f = join(save_dir, os.path.basename(image_f))
    print(f'Saving \"{save_f}\"')
    fig.savefig(save_f.replace('.jpg', '.png'), bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close('all')


def qual_3dpw(image_f, im_size=480, set_title=True):
    plt.rcParams["font.size"] = 20

    print(image_f)
    pare_f = image_f # join(PARE_DIR, image_f)
    spin_f = join(SPIN_DIR, image_f)
    hmr_f = join(HMR_DIR, image_f)

    input_image = pare_f.replace('output_images', 'output_meshes')

    pare_render = input_image.replace('.jpg', '_overlay.png')
    spin_render = pare_render.replace(PARE_DIR, SPIN_DIR)
    hmr_render = pare_render.replace(PARE_DIR, HMR_DIR)

    # if not os.path.isfile(pare_render):
    run_blender_on_single_obj(
        obj_f=pare_render.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size,
    )

    # if not os.path.isfile(spin_render):
    run_blender_on_single_obj(
        obj_f=spin_render.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size,
    )

    # if not os.path.isfile(hmr_render):
    run_blender_on_single_obj(
        obj_f=hmr_render.replace('_overlay.png', '.obj'), color='light_red', im_size=im_size,
    )

    os.makedirs('logs/figures/coco_v2_input_images', exist_ok=True)
    print(f'copying {input_image}')
    shutil.copy2(input_image, 'logs/figures/coco_v2_input_images')

    input_image = io.imread(input_image)
    spin_result = io.imread(spin_render)
    pare_result = io.imread(pare_render)
    hmr_result = io.imread(hmr_render)

    fig, ax = plt.subplots(
        ncols=4,
        figsize=(24, 9)
    )

    for i in range(4):
        ax[i].axis('off')

    if set_title:
        ax[0].set_title('Input Image')
        ax[1].set_title('SPIN')
        ax[2].set_title('PARE')

    ax[0].imshow(input_image)
    ax[1].imshow(spin_result)
    ax[2].imshow(hmr_result)
    ax[3].imshow(pare_result)
    fig.set_tight_layout(True)

    save_dir = 'logs/figures/coco_v2'
    os.makedirs(save_dir, exist_ok=True)
    fig.set_tight_layout(True)
    save_f = join(save_dir, os.path.basename(image_f))
    print(f'Saving \"{save_f}\"')
    fig.savefig(save_f.replace('.jpg', '.png'), bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close('all')


def run_blender_on_single_obj(obj_f, color, im_size, turntable=False):
    cmd = f'sh scripts/blender_render.sh {obj_f} {color} {im_size}'
    if turntable:
        cmd += ' --turntable'
    os.system(cmd)


def run_blender_on_folder(mesh_folder, color, im_size):
    cmd = f'sh scripts/blender_render.sh {mesh_folder} {color} {im_size}'
    os.system(cmd)


def teaser_figure_generator(img_file, im_size=480, row_1=False, occ_idx=None):
    fig, ax = plt.subplots(
        ncols=6,
        gridspec_kw={
            'width_ratios': [20,]*5 + [1,],
            'height_ratios': [1,],
            # 'wspace': 0.025,
            # 'hspace': 0.025,
        },
        figsize=(20,4)
    )

    pare_dir = join(PARE_DIR, 'occlusion_test_train')
    spin_dir = join(SPIN_DIR, 'logs/occlusion_test_train')

    input_image = join(pare_dir, 'output_meshes', img_file)

    if row_1:

        spin_result = input_image.replace(pare_dir, spin_dir).replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        pare_result = input_image.replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        # breakpoint()
        input_image = pare_result.replace('_overlay.png', '.jpg')

        if not os.path.isfile(spin_result):
            run_blender_on_single_obj(
                obj_f=spin_result.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size,
            )

        if not os.path.isfile(pare_result):
            run_blender_on_single_obj(
                obj_f=pare_result.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size,
            )
    else:
        spin_result = input_image.replace(pare_dir, spin_dir).replace('.jpg', '_overlay.png')
        pare_result = input_image.replace('.jpg', '_overlay.png')

    heatmap_color, heatmap_orig = normalize_heatmaps(
        method_1_dir=pare_dir, method_2_dir=spin_dir,
        img=img_file, joint_id=0, im_size=480,
    )

    input_image = io.imread(input_image)
    spin_result = io.imread(spin_result)
    pare_result = io.imread(pare_result)
    f = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    # plt.imshow(np.hstack([input_image, input_image]))
    # plt.imshow(f, alpha=0.5, cmap='jet')
    # clb = plt.colorbar()

    for i in range(5):
        ax[i].axis('off')

    ax[0].set_title('(a) Input Image')
    ax[0].imshow(input_image)
    ax[1].set_title('(b) SPIN')
    ax[1].imshow(spin_result)
    ax[2].set_title('(c) PARE')
    ax[2].imshow(pare_result)
    ax[3].set_title('(d) SPIN Error Heatmap')
    ax[3].imshow(input_image)
    ax[3].imshow(heatmap_color[1], alpha=ALPHA)
    ax[4].set_title('(e) PARE Error Heatmap')
    ax[4].imshow(input_image)
    ax[4].imshow(heatmap_color[0], alpha=ALPHA)

    heatmap_orig = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=heatmap_orig.min(), vmax=heatmap_orig.max())
    cb = mpl.colorbar.ColorbarBase(ax[5], cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Mean joint error (mm)')

    fig.set_tight_layout(True)

    save_f = join('logs/figures/teaser', img_file.replace('.jpg', '_teaser.png'))

    if row_1:
        save_f = save_f.replace('.png', '_row_1.png')
    else:
        save_f = save_f.replace('.png', '_row_2.png')

    print(f'Saving \"{save_f}\"')
    fig.savefig(save_f, bbox_inches='tight')
    # plt.show()
    plt.close('all')
    # pass


@plt.style.context('dark_background')
def supmat_video_teaser(img_file, im_size=480, occ_idxs=None):
    pare_dir = join(PARE_DIR, 'occlusion_test_train')
    spin_dir = join(SPIN_DIR, 'logs/occlusion_test_train')

    input_image = join(pare_dir, 'output_meshes', img_file)

    spin_results = []
    pare_results = []

    patch_coords = []

    # render the occlusion objs
    spin_meshes_folder = input_image.replace(pare_dir, spin_dir).replace('output_meshes', 'output_images').replace(
        '.jpg', f'_meshes/')
    pare_meshes_folder = input_image.replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/')

    # run_blender_on_folder(spin_meshes_folder, color='light_green', im_size=im_size)
    # run_blender_on_folder(pare_meshes_folder, color='turkuaz', im_size=im_size)

    for occ_idx in occ_idxs[:2]:
        spin_result = input_image.replace(pare_dir, spin_dir).replace('output_meshes', 'output_images').replace('.jpg',
                                                                                                                f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        pare_result = input_image.replace('output_meshes', 'output_images').replace('.jpg',
                                                                                    f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        # breakpoint()

        if not os.path.isfile(spin_result):
            run_blender_on_single_obj(
                obj_f=spin_result.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size, turntable=False
            )

        if not os.path.isfile(pare_result):
            run_blender_on_single_obj(
                obj_f=pare_result.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size, turntable=False
            )

        spin_results.append(spin_result)
        pare_results.append(pare_result)
        patch_coords.append([int(occ_idx / 11), occ_idx % 11])

    # generate video of these samples

    heatmap_color, heatmap_orig = normalize_heatmaps(
        method_1_dir=pare_dir, method_2_dir=spin_dir,
        img=img_file, joint_id=0, im_size=480,
    )

    input_image = io.imread(input_image)
    # spin_result = io.imread(spin_result)
    # pare_result = io.imread(pare_result)
    f = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    # plt.imshow(np.hstack([input_image, input_image]))
    # plt.imshow(f, alpha=0.5, cmap='jet')
    # clb = plt.colorbar()

    # 'width_ratios': [20, ] * 5 + [1, ],
    # 'height_ratios': [1, ],

    plt.rcParams['font.size'] = 11
    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[20, 10, 10, 10, 1], height_ratios=[1, 1])
    ax_inp_image = fig.add_subplot(gs[:, 0])
    ax_inp_image.axis('off')
    ax_spin_1 = fig.add_subplot(gs[0, 1])
    ax_spin_1.axis('off')
    ax_spin_2 = fig.add_subplot(gs[0, 2])
    ax_spin_2.axis('off')
    ax_spin_h = fig.add_subplot(gs[0, 3])
    ax_spin_h.axis('off')
    ax_pare_1 = fig.add_subplot(gs[1, 1])
    ax_pare_1.axis('off')
    ax_pare_2 = fig.add_subplot(gs[1, 2])
    ax_pare_2.axis('off')
    ax_pare_h = fig.add_subplot(gs[1, 3])
    ax_pare_h.axis('off')
    ax_colbar = fig.add_subplot(gs[:, -1])

    # for i in range(5):
    #     ax[i].axis('off')

    ax_inp_image.set_title('(a) Input Image', fontsize=20)
    ax_inp_image.imshow(input_image)
    ax_spin_1.set_title('(b) SPIN [1]', fontsize=20)
    ax_spin_1.imshow(io.imread(spin_results[0]))
    ax_spin_2.set_title('(c) SPIN [2]', fontsize=20)
    ax_spin_2.imshow(io.imread(spin_results[1]))
    ax_spin_h.set_title('SPIN Occlusion Sensitivity Heatmap')
    ax_spin_h.imshow(input_image)
    ax_spin_h.imshow(heatmap_color[1], alpha=ALPHA)

    # print(patch_coords)
    # for idx, coord in enumerate(patch_coords):
    #     s_patch = 40 * (im_size / 224)
    #     s_move = 20 * (im_size / 224)
    #     c1, c2 = s_move * coord[0], s_move * coord[1]
    #     color = 'w'  # if idx == 0 else 'r'
    #     rect = patches.Rectangle((c2, c1), s_patch, s_patch, linewidth=2, edgecolor=color, facecolor='none')
    #     # Add the patch to the Axes
    #     ax_spin_h.add_patch(rect)
    #     ax_spin_h.text(c2 - 30, c1 + 25, f'{idx + 1}', color=color, fontsize=20)

    ax_pare_1.set_title('(e) PARE [1]', fontsize=20)
    ax_pare_1.imshow(io.imread(pare_results[0]))
    ax_pare_2.set_title('(f) PARE [2]', fontsize=20)
    ax_pare_2.imshow(io.imread(pare_results[1]))
    ax_pare_h.set_title('PARE Occlusion Sensitivity Heatmap')
    ax_pare_h.imshow(input_image)
    ax_pare_h.imshow(heatmap_color[0], alpha=ALPHA)

    # for idx, coord in enumerate(patch_coords):
    #     s_patch = 40 * (im_size / 224)
    #     s_move = 20 * (im_size / 224)
    #     c1, c2 = s_move * coord[0], s_move * coord[1]
    #     color = 'w'  # if idx == 0 else 'r'
    #     rect = patches.Rectangle((c2, c1), s_patch, s_patch, linewidth=2, edgecolor=color, facecolor='none')
    #     # Add the patch to the Axes
    #     ax_pare_h.add_patch(rect)
    #
    #     ax_pare_h.text(c2 - 30, c1 + 30, f'{idx + 1}', color=color, fontsize=20)

    heatmap_orig = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=heatmap_orig.min(), vmax=heatmap_orig.max())
    cb = mpl.colorbar.ColorbarBase(ax_colbar, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Mean 3D joint error (mm)', fontsize=20)

    # fig.set_tight_layout(True)

    save_f = join('logs/figures/supmat_teaser', img_file.replace('.jpg', f'_teaser{FIG_EXT}'))
    os.makedirs(os.path.dirname(save_f), exist_ok=True)
    print(f'Saving \"{save_f}\"')
    io.imsave(save_f.replace('teaser.pdf', 'pare_hm.png'), overlay_heatmaps(input_image, heatmap_color[0], alpha=0.5))
    io.imsave(save_f.replace('teaser.pdf', 'spin_hm.png'), overlay_heatmaps(input_image, heatmap_color[1], alpha=0.5))
    # fig.savefig(save_f, bbox_inches='tight', transparent=True)

    save_f = join('logs/figures/supmat_teaser', img_file.replace('.jpg', f'_teaser.png'))
    fig.savefig(save_f, bbox_inches='tight', transparent=True)
    # plt.show()
    img = io.imread(save_f)
    img = img[:,1360:,:]
    io.imsave(save_f, img)
    plt.close('all')
    # pass


# @plt.style.context('dark_background')
def teaser_figure_generator_v2(img_file, im_size=480, occ_idxs=None):

    pare_dir = join(PARE_DIR, 'occlusion_test_train')
    spin_dir = join(SPIN_DIR, 'logs/occlusion_test_train')

    input_image = join(pare_dir, 'output_meshes', img_file)

    spin_results = []
    pare_results = []

    patch_coords = []

    # render the occlusion objs
    spin_meshes_folder = input_image.replace(pare_dir, spin_dir).replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/')
    pare_meshes_folder = input_image.replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/')

    # run_blender_on_folder(spin_meshes_folder, color='light_green', im_size=im_size)
    # run_blender_on_folder(pare_meshes_folder, color='turkuaz', im_size=im_size)

    for occ_idx in occ_idxs[:2]:
        spin_result = input_image.replace(pare_dir, spin_dir).replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        pare_result = input_image.replace('output_meshes', 'output_images').replace('.jpg', f'_meshes/result_00_{occ_idx:05d}_overlay.png')
        # breakpoint()

        if not os.path.isfile(spin_result):
            run_blender_on_single_obj(
                obj_f=spin_result.replace('_overlay.png', '.obj'), color='light_green', im_size=im_size, turntable=False
            )

        if not os.path.isfile(pare_result):
            run_blender_on_single_obj(
                obj_f=pare_result.replace('_overlay.png', '.obj'), color='turkuaz', im_size=im_size, turntable=False
            )

        spin_results.append(spin_result)
        pare_results.append(pare_result)
        patch_coords.append([int(occ_idx / 11), occ_idx % 11])

    # generate video of these samples

    heatmap_color, heatmap_orig = normalize_heatmaps(
        method_1_dir=pare_dir, method_2_dir=spin_dir,
        img=img_file, joint_id=0, im_size=480,
    )

    input_image = io.imread(input_image)
    # spin_result = io.imread(spin_result)
    # pare_result = io.imread(pare_result)
    f = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    # plt.imshow(np.hstack([input_image, input_image]))
    # plt.imshow(f, alpha=0.5, cmap='jet')
    # clb = plt.colorbar()

    # 'width_ratios': [20, ] * 5 + [1, ],
    # 'height_ratios': [1, ],

    plt.rcParams['font.size'] = 11
    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[20,10,10,10,1], height_ratios=[1,1])
    ax_inp_image = fig.add_subplot(gs[:, 0])
    ax_inp_image.axis('off')
    ax_spin_1 = fig.add_subplot(gs[0, 1])
    ax_spin_1.axis('off')
    ax_spin_2 = fig.add_subplot(gs[0, 2])
    ax_spin_2.axis('off')
    ax_spin_h = fig.add_subplot(gs[0, 3])
    ax_spin_h.axis('off')
    ax_pare_1 = fig.add_subplot(gs[1, 1])
    ax_pare_1.axis('off')
    ax_pare_2 = fig.add_subplot(gs[1, 2])
    ax_pare_2.axis('off')
    ax_pare_h = fig.add_subplot(gs[1, 3])
    ax_pare_h.axis('off')
    ax_colbar = fig.add_subplot(gs[:, -1])

    # for i in range(5):
    #     ax[i].axis('off')

    ax_inp_image.set_title('(a) Input Image', fontsize=20)
    ax_inp_image.imshow(input_image)
    ax_spin_1.set_title('(b) SPIN [1]', fontsize=20)
    ax_spin_1.imshow(io.imread(spin_results[0]))
    ax_spin_2.set_title('(c) SPIN [2]', fontsize=20)
    ax_spin_2.imshow(io.imread(spin_results[1]))
    ax_spin_h.set_title('(d) SPIN Occlusion Sensitivity Heatmap')
    ax_spin_h.imshow(input_image)
    ax_spin_h.imshow(heatmap_color[1], alpha=ALPHA)

    print(patch_coords)
    for idx, coord in enumerate(patch_coords):
        s_patch = 40 * (im_size / 224)
        s_move = 20 * (im_size / 224)
        c1, c2 = s_move * coord[0], s_move * coord[1]
        color = 'w' # if idx == 0 else 'r'
        rect = patches.Rectangle((c2, c1), s_patch, s_patch, linewidth=2, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax_spin_h.add_patch(rect)
        ax_spin_h.text(c2-30, c1+25, f'{idx+1}', color=color, fontsize=20)


    ax_pare_1.set_title('(e) PARE [1]', fontsize=20)
    ax_pare_1.imshow(io.imread(pare_results[0]))
    ax_pare_2.set_title('(f) PARE [2]', fontsize=20)
    ax_pare_2.imshow(io.imread(pare_results[1]))
    ax_pare_h.set_title('(g) PARE Occlusion Sensitivity Heatmap')
    ax_pare_h.imshow(input_image)
    ax_pare_h.imshow(heatmap_color[0], alpha=ALPHA)

    for idx, coord in enumerate(patch_coords):
        s_patch = 40 * (im_size / 224)
        s_move = 20 * (im_size / 224)
        c1, c2 = s_move * coord[0], s_move * coord[1]
        color = 'w' # if idx == 0 else 'r'
        rect = patches.Rectangle((c2, c1), s_patch, s_patch, linewidth=2, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax_pare_h.add_patch(rect)

        ax_pare_h.text(c2-30, c1+30, f'{idx+1}', color=color, fontsize=20)

    heatmap_orig = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=heatmap_orig.min(), vmax=heatmap_orig.max())
    cb = mpl.colorbar.ColorbarBase(ax_colbar, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Mean 3D joint error (mm)', fontsize=20)

    # fig.set_tight_layout(True)

    save_f = join('logs/figures/teaser_v2', img_file.replace('.jpg', f'_teaser{FIG_EXT}'))
    os.makedirs(os.path.dirname(save_f), exist_ok=True)
    print(f'Saving \"{save_f}\"')
    io.imsave(save_f.replace('teaser.pdf', 'pare_hm.png'), overlay_heatmaps(input_image, heatmap_color[0], alpha=0.5))
    io.imsave(save_f.replace('teaser.pdf', 'spin_hm.png'), overlay_heatmaps(input_image, heatmap_color[1], alpha=0.5))
    # fig.savefig(save_f, bbox_inches='tight', transparent=True)

    save_f = join('logs/figures/teaser_v2', img_file.replace('.jpg', f'_teaser.png'))
    fig.savefig(save_f, bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close('all')
    # pass


def generate_occlusion_videos(img_file, occ_idxs, type='spin'):
    # cmd = 'ffmpeg -y -framerate 8 -i /ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/pare_coco/05.11-spin_ckpt_eval/logs/occlusion_test_train/output_images/result_00_26600_meshes/result_00_%05d.jpg -codec:v libx264 logs/figures/patch.mp4'

    pare_dir = join(PARE_DIR, 'occlusion_test_train')
    spin_dir = join(SPIN_DIR, 'logs/occlusion_test_train')

    dir = spin_dir if type == 'spin' else pare_dir

    dest = 'logs/figures/supmat_video_material/occlusion'

    video_dir = join(dir, 'output_images', img_file.replace('.jpg', '_meshes'))

    out_vid_name = img_file.replace('.jpg', f'_{type}_input.mp4')
    os.system(f'ffmpeg -y -framerate 8 -i {video_dir}/result_00_%05d.jpg -codec:v libx264 {dest}/{out_vid_name}')

    out_vid_name = img_file.replace('.jpg', f'_{type}_overlay.mp4')
    os.system(f'ffmpeg -y -framerate 8 -i {video_dir}/result_00_%05d_overlay.png -codec:v libx264 {dest}/{out_vid_name}')

    for occ_idx in occ_idxs[:2]:
        video_dir = join(dir, 'output_images', img_file.replace('.jpg', '_meshes'), f'result_00_{occ_idx:05d}')
        dst_f_name = img_file.replace('.jpg', f'_{type}_{occ_idx:05d}')
        print(f'Copying {dst_f_name}')
        os.system(f'cp -r {video_dir} {dest}/{dst_f_name}')


def copy_qual_images_to_dropbox(type='coco', idx=-1):

    def collate_images(img, overlay, table, dest):

        for cfg in [1,2,3]:
            os.makedirs(f'{dest}_{cfg}', exist_ok=True)


        img = io.imread(img)
        overlay = io.imread(overlay)

        ii = np.concatenate([img, overlay], axis=1)
        io.imsave(f'{dest}_2/000.png', ii)

        turntable_imgs = [x for x in sorted(os.listdir(table))]
        for idx, t in enumerate(turntable_imgs):
            t = io.imread(os.path.join(table, t))[:,:,:3]

            ii = np.concatenate([overlay, t], axis=1)
            io.imsave(f'{dest}_3/{idx:03d}.png', ii)

            ii = np.concatenate([img, ii], axis=1)
            io.imsave(f'{dest}_1/{idx:03d}.png', ii)

    if type == 'coco':
        with open(join(PARE_DIR, 'evaluation_coco_mpii/good_image_samples.txt'), 'r') as f:
            img_files = [x.rstrip() for x in f.readlines()]

        dest = 'logs/figures/supmat_video_material/coco_qual_results'
        # pare_dest = '/home/mkocabas/Dropbox/PARE-CVPR2021/SupMat/video_project/coco_qual_results/pare'
        # hmr_dest  = '/home/mkocabas/Dropbox/PARE-CVPR2021/SupMat/video_project/coco_qual_results/hmr'
    elif type == '3dpw':
        with open(join(PARE_DIR, 'evaluation_3dpw-all_mpi-inf-3dhp/good_image_samples.txt'), 'r') as f:
            img_files = [x.rstrip() for x in f.readlines()]

        dest = 'logs/figures/supmat_video_material/3dpw_qual_results'
        # pare_dest = 'logs/figures/supmat_video_material/3dpw/pare'
        # hmr_dest = 'logs/figures/supmat_video_material/3dpw/hmr'
    else:
        exit()

    os.makedirs(dest, exist_ok=True)
    # os.makedirs(pare_dest, exist_ok=True)
    # os.makedirs(hmr_dest, exist_ok=True)
    if idx > -1:
        img_files = img_files[idx:idx+1]

    for image_f in img_files:
        print(os.path.basename(image_f))
        input_image = image_f.replace('output_images', 'output_meshes')
        pare_f = input_image
        spin_f = input_image
        hmr_f = input_image

        pare_render = input_image.replace('.jpg', '_overlay.png')
        spin_render = pare_render.replace(PARE_DIR, SPIN_DIR)
        hmr_render = pare_render.replace(PARE_DIR, HMR_DIR)

        pare_table = pare_render.replace('_overlay.png', '')
        spin_table = spin_render.replace('_overlay.png', '')
        hmr_table = hmr_render.replace('_overlay.png', '')

        collate_images(pare_f, pare_render, pare_table, dest=f'{dest}/{os.path.basename(image_f)}_pare')
        collate_images(spin_f, spin_render, spin_table, dest=f'{dest}/{os.path.basename(image_f)}_spin')
        collate_images(hmr_f, hmr_render, hmr_table, dest=f'{dest}/{os.path.basename(image_f)}_hmr')


def run_demo_on_videos(idx):
    videos = [
        'friends_bradpitt_0.25-0.37.mp4',
        'matrix_kungfu.mp4',
        'first_aid_office.mp4',
        'VIP_videos6.mp4',
        'eliud.mp4',
        'ballet_high_quality.mp4',
        'outdoors_freestyle_00.mp4',
        'friends_red_sweater.mp4',
        'MOT17-08-FRCNN.mp4',
        'the_greatest_showman_dance.mp4',
        'kevin_durant_slomo_part2.mp4',
        'VIP_videos285.mp4',
        'outdoors_crosscountry_00.mp4',
        'outdoors_fencing_01.mp4',
        'las_vegas_dance.mp4',
        'friends_bradpitt.mp4',
        'radical_dance.webm',
        'dance_cmu.mp4',
        'flat_packBags_00.mp4',
        'outdoors_slalom_00.mp4',
        'dimitri_dance.mp4',
        'VIP_videos50.mp4',
        'jackie_chan.webm',
        'salsa.mp4',
        'mission_impossible.mp4',
        'atlas.mp4',
        'VIP_videos260.mp4',
        'fire_drill_office.mp4',
        'outdoors_parcours_00.mp4',
        'best_dunks.mp4',
        'bowing_thank_you.mp4',
        'sydney_harbor_dance.mp4',
        'BasementSittingBooth_03403_01.mp4',
        'friends_mesh.mp4',
        'VIP_videos49.mp4',
        'spectre4k.mp4',
        'kevin_durant_slomo_part1.mp4',
        'swan_lake.mp4',
        'spectre.mp4'
        'kosu.mp4',
        'sample_video.mp4',
     ]

    methods = ['pare', 'spin', 'eft', 'vibe']
    videos_dir = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/demo_results/videos'
    input_video, method = list(itertools.product(videos, methods))[idx]

    video_path = join(videos_dir, input_video)
    output_path = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/demo_results'

    python_path = '/home/mkocabas/miniconda3/bin/python'

    cfg = ckpt = None
    exp = ''
    if method == 'pare':
        cfg = join(PARE_DIR, 'config_to_run.yaml')
        ckpt = join(PARE_DIR, 'tb_logs_pare/0_1111239d5702427e8088b30d8584f263/checkpoints/epoch=0.ckpt')
        exp = 'pare-hrnet'
    elif method == 'spin':
        cfg = join(SPIN_DIR, 'config_to_run.yaml')
        ckpt = 'spin'
        exp = 'spin'
    elif method == 'eft':
        cfg = join(HMR_DIR, 'config_to_run.yaml')
        ckpt = join(HMR_DIR, 'tb_logs_pare/0_2f2e90aa0c3242b095216e2f7ab556e4/checkpoints/epoch=6.ckpt')
        exp = 'eft'

    if method == 'vibe':
        os.chdir('/ps/scratch/mkocabas/developments/VIBE-tests-demo-old/')
        exp = 'vibe'
        cmd = f'{python_path} demo.py --exp {exp} ' \
              f'--vid_file {video_path} --output_folder {output_path}'
        print(cmd)
        os.system(cmd)
    else:
        cmd = f'{python_path} demo.py --cfg {cfg} --ckpt {ckpt} --exp {exp} ' \
              f'--vid_file {video_path} --output_folder {output_path}'
        print(cmd)
        os.system(cmd)


def create_mosaic(idx):
    videos = [
        'friends_bradpitt_0.25-0.37.mp4',
        'matrix_kungfu.mp4',
        'first_aid_office.mp4',
        'VIP_videos6.mp4',
        'eliud.mp4',
        'ballet_high_quality.mp4',
        'outdoors_freestyle_00.mp4',
        'friends_red_sweater.mp4',
        'MOT17-08-FRCNN.mp4',
        'the_greatest_showman_dance.mp4',
        'kevin_durant_slomo_part2.mp4',
        'VIP_videos285.mp4',
        'outdoors_crosscountry_00.mp4',
        'outdoors_fencing_01.mp4',
        'las_vegas_dance.mp4',
        'friends_bradpitt.mp4',
        'radical_dance.webm',
        'dance_cmu.mp4',
        'flat_packBags_00.mp4',
        'outdoors_slalom_00.mp4',
        'dimitri_dance.mp4',
        'VIP_videos50.mp4',
        'jackie_chan.webm',
        'salsa.mp4',
        'mission_impossible.mp4',
        'atlas.mp4',
        'VIP_videos260.mp4',
        'fire_drill_office.mp4',
        'outdoors_parcours_00.mp4',
        'best_dunks.mp4',
        'bowing_thank_you.mp4',
        'sydney_harbor_dance.mp4',
        'BasementSittingBooth_03403_01.mp4',
        'friends_mesh.mp4',
        'VIP_videos49.mp4',
        'spectre4k.mp4',
        'kevin_durant_slomo_part1.mp4',
        'swan_lake.mp4',
        'spectre.mp4'
        'kosu.mp4',
        'sample_video.mp4',
    ]

    v = os.path.basename(videos[idx]).replace('.mp4', '')
    cmd = f'sh tests/test_ffmpeg_mosaic.sh {v}'
    os.system(cmd)


def save_colors():
    import colorsys
    for m in ['pare', 'spin', 'eft', '']:
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in range(10000)}
        joblib.dump(mesh_color, f'data/demo_mesh_colors_{m}.npy')


if __name__ == '__main__':
    mode = sys.argv[1]

    if 'att' in mode:
        attention_fig(debug=True)
        # attention_fig(debug=True, quantize=True)

    if 'row1' in mode:
        # prepare row_2 results
        with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_occlusion_failures.txt') as f:
            for x in f.readlines():
                img_file = x.rstrip().split(',')[0]
                occ_idx = int(x.rstrip().split(',')[1])
                teaser_figure_generator(img_file=img_file, row_1=True, occ_idx=occ_idx)

    if 'row2' in mode:
        # prepare row_2 results
        with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_failure.txt') as f:
            for x in f.readlines():
                teaser_figure_generator(img_file=x.rstrip())


    if 'coco' in mode:
        if len(sys.argv) > 2:
            with open(join(PARE_DIR, 'evaluation_coco_mpii/good_image_samples.txt'), 'r') as f:
                img_files = [x.rstrip() for x in f.readlines()]
            idx = int(sys.argv[2])
            print('Processing', idx, img_files[idx])
            qual_coco(img_files[idx], set_title=False)
        else:
            with open(join(PARE_DIR, 'evaluation_coco_mpii/good_image_samples.txt'), 'r') as f:
                for x in f.readlines():
                    qual_coco(x.rstrip(), set_title=False)

    if 'fail' in mode:
        with open(join(PARE_DIR, 'evaluation_coco_mpii/bad_image_samples.txt'), 'r') as f:
            for x in f.readlines():
                qual_coco_fail(x.rstrip(), set_title=False)

    if '3dpw' in mode:
        if len(sys.argv) > 2:
            with open(join(PARE_DIR, 'evaluation_3dpw-all_mpi-inf-3dhp/good_image_samples.txt'), 'r') as f:
                img_files = [x.rstrip() for x in f.readlines()]
            idx = int(sys.argv[2])
            print('Processing', idx, img_files[idx])
            qual_coco(img_files[idx], set_title=False, output_folder='3dpw_qual')
        else:
            with open(join(PARE_DIR, 'evaluation_3dpw-all_mpi-inf-3dhp/good_image_samples.txt'), 'r') as f:
                for x in f.readlines():
                    qual_coco(x.rstrip(), set_title=False, output_folder='3dpw_qual')


    if 'teaser' in mode:
        # prepare row_2 results
        if len(sys.argv) > 2:
            with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_failure_double.txt') as f:
                img_files = []
                occ_idxs = []
                for x in f.readlines():
                    img_file = x.rstrip().split(',')[0]
                    occ_idx = [int(x) for x in x.rstrip().split(',')[1:]]

                    img_files.append(img_file)
                    occ_idxs.append(occ_idx)

            idx = int(sys.argv[2])
            print('Processing', idx, img_files[idx])
            teaser_figure_generator_v2(img_file=img_files[idx], occ_idxs=occ_idxs[idx])
        else:
            with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_failure_double.txt') as f:
                for x in f.readlines():
                    img_file = x.rstrip().split(',')[0]
                    occ_idx = [int(x) for x in x.rstrip().split(',')[1:]]
                    teaser_figure_generator_v2(img_file=img_file, occ_idxs=occ_idx)

    if 'occ' in mode:
        occlusion_meshes()

    if 'sens' in mode:
        occlusion_sensitivity_hm()

    if 'copy' in mode:
        copy_qual_images_to_dropbox(type='3dpw', idx=int(sys.argv[2]))

    if 'mk' in mode:
        with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_failure_double.txt') as f:
            for x in f.readlines():
                img_file = x.rstrip().split(',')[0]
                occ_idx = [int(x) for x in x.rstrip().split(',')[1:]]
                generate_occlusion_videos(img_file=img_file, occ_idxs=occ_idx, type='pare')

    if 'demo' in mode:
        run_demo_on_videos(idx=int(sys.argv[2]))

    if 'color' in mode:
        save_colors()

    if 'mosaic' in mode:
        create_mosaic(idx=int(sys.argv[2]))

    if 'sup_teas' in mode:
        with open('logs/pare_coco/05.11-spin_ckpt_eval/spin_failure_double.txt') as f:
            for x in f.readlines():
                img_file = x.rstrip().split(',')[0]
                occ_idx = [int(x) for x in x.rstrip().split(',')[1:]]
                supmat_video_teaser(img_file=img_file, occ_idxs=occ_idx)
