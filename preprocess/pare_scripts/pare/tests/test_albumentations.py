import os
import cv2
import random
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
from pare.core.config import DATASET_FILES, DATASET_FOLDERS
from pare.utils.image_utils import crop, transform

KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y, v) in keypoints:
        if v > 0:
            cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    # plt.axis('off')
    plt.imshow(image)


def j2d_processing(kp, center, scale, r, res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i,0:2] = transform(kp[i,0:2] + 1, center, scale, [res, res], rot=r)
    # convert to normalized coordinates
    # kp[:,:-1] = 2. *kp[:,:-1] / res - 1.
    # flip the x coordinates

    kp = kp.astype('float32')
    return kp

if __name__ == '__main__':

    data = np.load(DATASET_FILES[1]['mpii'])
    img_dir = DATASET_FOLDERS['mpii']


    for idx in range(50):
        imgname = os.path.join(img_dir, data['imgname'][idx])
        image = cv2.imread(imgname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = data['part'][idx]
        center = data['center'][idx]
        scale = data['scale'][idx]
        print('center', center, 'scale', scale)
        bbox = np.array([center[0], center[1], scale, scale])
        ul = (center.T - (scale / 2.)).astype(int)
        br = (center.T + (scale / 2.)).astype(int)

        orig_res = 224 # int(round(scale*200))
        crop_scale = 0.3
        # vis_keypoints(image, keypoints)
        image = crop(img=image, center=center, scale=scale, res=[orig_res, orig_res])
        # keypoints = j2d_processing(keypoints, center, scale, r=0, res=orig_res)
        transform_alb = A.Compose(
            [
                # A.Crop(x_min=ul[0], y_min=ul[1], x_max=br[0], y_max=br[1], p=1.0),
                # A.Rotate(limit=45, p=1.0),
                A.RandomCrop(width=int(orig_res*crop_scale), height=int(orig_res*crop_scale), p=1)
            ],
            # keypoint_params=A.KeypointParams(format='xy')
        )
        transformed = transform_alb(image=image) # , keypoints=keypoints)
        # vis_keypoints(transformed['image'], transformed['keypoints'])
        plt.imshow(transformed['image'])
        plt.show()