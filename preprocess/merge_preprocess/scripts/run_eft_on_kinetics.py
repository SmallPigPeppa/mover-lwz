import os
import cv2
import sys
import time
import torch
import shutil
import joblib
import argparse
import colorsys
import skvideo.io
import numpy as np
import os.path as osp
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pare.eft.eft_fitter import EFTFitter
from pare.utils.kp_utils import convert_kps
from pare.dataset.inference import Inference
from pare.eft.config import get_cfg_defaults
from pare.utils.vibe_renderer import Renderer
from pare.utils.vis_utils import  draw_skeleton
from pare.utils.smooth_pose import smooth_pose
from pare.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 300
MIN_NUM_KP = 15
MIN_THRESHOLD = 0.7
MIN_NUM_PERSON = 6

vid_folder = '/ps/scratch/mkocabas/data/Kinetics-400/videos'
folder = '/ps/scratch/mkocabas/data/Kinetics-400/posetrack_annotations'


def main(args):
    ann_files = joblib.load('/ps/scratch/mkocabas/data/Kinetics-400/kinetics_valid_annotations_1.8M.pkl')
    np.random.shuffle(ann_files)

    logger.info('Number of videos', len(ann_files))

    hparams = get_cfg_defaults()
    hparams.LOG = False
    hparams.LOSS.OPENPOSE_TRAIN_WEIGHT = 1.0

    device = 'cuda' if torch.cuda.is_available() else sys.exit('CUDA is not available')

    eft_fitter = EFTFitter(hparams=hparams)

    # for ann in ann_files[:args.num_videos]:
    #     print(ann)
    # exit()

    ann = args.ann
    # for ann in ann_files[:args.num_videos]:

    logger.info(ann)

    action, fn, person_id = ann.split('/')
    person_id = int(person_id)
    data = joblib.load(osp.join(folder, action, fn))

    video_file = osp.join(vid_folder, action, fn.replace('.pkl', '.mp4'))
    # vf = skvideo.io.vread(video_file).astype(float)

    log_dir = os.path.join('logs/eft/kinetics', video_file.split('/')[-1])

    if os.path.isdir(os.path.join(log_dir, 'tmp_images')):
        image_folder = os.path.join(log_dir, 'tmp_images')
        logger.info(f'Frames are already extracted in \"{image_folder}\"')
        num_frames = len(os.listdir(image_folder))
        img_shape = cv2.imread(os.path.join(image_folder, '000001.png')).shape
    else:
        image_folder, num_frames, img_shape = video_to_images(
            video_file,
            img_folder=os.path.join(log_dir, 'tmp_images'),
            return_info=True
        )

    orig_height, orig_width = img_shape[:2]

    joints2d = convert_kps(data[person_id]['joints2d'], src='staf', dst='spin')

    dataset = Inference(
        image_folder=image_folder,
        frames=data[person_id]['frames'],
        bboxes=None,
        joints2d=joints2d,
        scale=1.0,
        return_dict=True,
        crop_size=hparams.DATASET.IMG_RES,
        normalize_kp2d=True,
    )

    dataloader = DataLoader(dataset, batch_size=hparams.DATASET.BATCH_SIZE, num_workers=8)

    bboxes = dataset.bboxes
    frames = dataset.frames
    # has_keypoints = True if joints2d is not None else False

    pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
    vibe_results = {}

    for batch_nb, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = eft_fitter.finetune_step(batch, batch_nb)

        pred_cam.append(output['pred_cam'])
        pred_verts.append(output['smpl_vertices'])
        pred_pose.append(output['pred_pose'])
        pred_betas.append(output['pred_shape'])
        pred_joints3d.append(output['smpl_joints3d'])

    pred_cam = torch.cat(pred_cam, dim=0)
    pred_verts = torch.cat(pred_verts, dim=0)
    pred_pose = torch.cat(pred_pose, dim=0)
    pred_betas = torch.cat(pred_betas, dim=0)
    pred_joints3d = torch.cat(pred_joints3d, dim=0)

    del batch

    with torch.no_grad():
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        if args.smooth:
            min_cutoff = args.min_cutoff  # 0.004
            beta = args.beta  # 1.5
            logger.info(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bboxes,
        img_width=orig_width,
        img_height=orig_height
    )

    output_dict = {
        'pred_cam': pred_cam,
        'orig_cam': orig_cam,
        'verts': pred_verts,
        'pose': pred_pose,
        'betas': pred_betas,
        'joints3d': pred_joints3d,
        'joints2d': joints2d,
        'bboxes': bboxes,
        'frame_ids': frames,
    }

    vibe_results[person_id] = output_dict

    # ========= Render results as a single video ========= #
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

    output_img_folder = f'{image_folder}_output'
    os.makedirs(output_img_folder, exist_ok=True)

    logger.info(f'Rendering output video, writing frames to {output_img_folder}')

    # prepare results for rendering
    frame_results = prepare_rendering_results(vibe_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)

        side_img = np.zeros_like(img)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mc = mesh_color[person_id]

            mesh_filename = None

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )

            side_img = renderer.render(
                side_img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                angle=270,
                axis=[0, 1, 0],
            )

        img = np.concatenate([img, side_img], axis=1)

        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

        if args.display:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    # ========= Save rendered video ========= #
    vid_name = os.path.basename(video_file)
    if args.smooth:
        save_name = f'{vid_name.replace(".mp4", "")}_smooth_eft_result.mp4'
    else:
        save_name = f'{vid_name.replace(".mp4", "")}_eft_result.mp4'

    save_name = os.path.join(log_dir, save_name)
    logger.info(f'Saving result video to {save_name}')
    images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
    if args.delete:
        shutil.rmtree(output_img_folder)

    shutil.rmtree(image_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', type=str, required=True,
                        help='ann file')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')
    parser.add_argument('--delete', action='store_true',
                        help='delete tmp files after run')
    parser.add_argument('--num_videos', default=25, type=int,
                        help='number of videos to render')
    # 1 EURO FILTER
    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')
    parser.add_argument('--min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    args = parser.parse_args()

    main(args)