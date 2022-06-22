import os
import cv2
import argparse
import numpy as np
import os.path as osp
from loguru import logger
import matplotlib.pyplot as plt
from skimage.io import imread, imsave


def methods_wo_error(method_1_dir, method_2_dir):
    # white_strip = np.ones([484, 100, 3]) * 255

    # n_images = len(os.listdir(osp.join(method_1_dir, 'output_images')))

    image_files = [osp.join(method_1_dir, 'output_images', x)
                   for x in sorted(os.listdir(osp.join(method_1_dir, 'output_images')))]
    # analysis_dir = osp.join(method_1_dir, 'analysis')
    # os.makedirs(analysis_dir, exist_ok=True)

    good_samples_f = osp.join(method_1_dir, 'good_image_samples.txt')
    f = open(good_samples_f, 'w')
    f.close()

    for id, img_f in enumerate(image_files):
        logger.info(f'Image id: {id}')

        im_method_1 = imread(img_f) # osp.join(method_1_dir, 'output_images', f'result_{id:05d}.jpg'))
        im_method_2 = imread(img_f.replace(method_1_dir, method_2_dir)) #osp.join(method_2_dir, 'output_images', f'result_{id:05d}.jpg'))
        im = np.hstack([im_method_1, im_method_2])
        # imsave(osp.join(analysis_dir, f'{id:05d}.jpg'), im)

        cv2.imshow('img', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        choice = input('click "y" if the sample is good:')
        if choice == 'y':
            print(img_f)
            with open(good_samples_f, 'a') as f:
                f.write(f'{img_f}\n')
        else:
            continue


def methods_with_error(method_1_dir, method_2_dir, n_images=20, show=False, save=True, use_mpjpe=1):
    method_1_results = np.load(osp.join(method_1_dir, 'val_images_error.npy')) * 1000
    method_2_results = np.load(osp.join(method_2_dir, 'val_images_error.npy')) * 1000

    dim_comp = use_mpjpe
    diff = method_1_results[:, dim_comp] - method_2_results[:, dim_comp]

    method_1_imgf = sorted(
        [osp.join(method_1_dir, 'output_images', x) for x in os.listdir(osp.join(method_1_dir, 'output_images')) if x.endswith('.jpg')]
    )
    method_2_imgf = sorted(
        [osp.join(method_2_dir, 'output_images', x) for x in os.listdir(osp.join(method_2_dir, 'output_images')) if x.endswith('.jpg')]
    )

    good_samples_f = osp.join(method_1_dir, 'good_image_samples.txt')

    # import IPython; IPython.embed(); exit(1)

    # white_strip = np.ones([imread(method_1_imgf[0]).shape[0], 100, 3]) * 255

    analysis_dir = osp.join(method_1_dir, 'analysis')
    os.makedirs(osp.join(analysis_dir, 'worse'), exist_ok=True)
    os.makedirs(osp.join(analysis_dir, 'best'), exist_ok=True)

    # plt.figure(figsize=(20, 40))
    # plt.tight_layout()
    for step, (id, derr) in enumerate(zip(np.argsort(diff), np.sort(diff))):
        # logger.info(f'{step} \t Image id: {id} \t Error diff: {derr}')
        # print(method_1_imgf[id])
        im_method_1 = imread(method_1_imgf[id]) # osp.join(method_1_dir, 'output_images', f'result_{id:05d}.jpg'))
        im_method_2 = imread(method_2_imgf[id]) # osp.join(method_2_dir, 'output_images', f'result_{id:05d}.jpg'))

        logger.info(f'{step} \t Image id: {id} \t Error diff: {derr} \t Img name: {os.path.basename(method_1_imgf[id])}')

        im = np.hstack([im_method_1, im_method_2])
        if show:
            cv2.imshow('err_img', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            choice = input('click "y" if the sample is good:')
            if choice == 'y':
                print(method_1_imgf[id])
                with open(good_samples_f, 'a') as f:
                    f.write(f'{method_1_imgf[id]}\n')
            else:
                continue

        elif save:
            imsave(osp.join(analysis_dir, 'best', f'{step:06d}.jpg'), im)

        if step == n_images: break

    for step, (id, derr) in enumerate(zip(np.argsort(diff)[::-1], np.sort(diff)[::-1])):
        logger.info(f'{step} \t Image id: {id} \t Error diff: {derr}')

        im_method_1 = imread(method_1_imgf[id])  # osp.join(method_1_dir, 'output_images', f'result_{id:05d}.jpg'))
        im_method_2 = imread(method_2_imgf[id])  # osp.join(method_2_dir, 'output_images', f'result_{id:05d}.jpg'))

        im = np.hstack([im_method_1, im_method_2])
        if show:
            cv2.imshow('img', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif save:
            imsave(osp.join(analysis_dir, 'worse', f'{step:06d}.jpg'), im)

        if step == n_images: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_1', type=str, help='method 1 dir')
    parser.add_argument('--method_2', type=str, help='method 2 dir')
    parser.add_argument('--nimg', type=int, default=20, help='num images to save or show')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    methods_with_error(args.method_1, args.method_2, n_images=args.nimg, save=args.save, show=args.show, use_mpjpe=1)
    # methods_wo_error(args.method_1, args.method_2)