import os
import sys
import cv2

sys.path.append('/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo')
from body_models.smplifyx.data_parser import read_keypoints
import glob
import json
import numpy as np
from tqdm import tqdm
def draw_op(kpts, img):
    print(kpts.shape)
    kpts = kpts.T
    # mean_k = kpts.mean(1)
    # kpts[:-1, :] = (kpts[:-1, :] - mean_k[:-1][:, None]) * 2 + mean_k[:-1][:, None]

    tkpts = np.zeros((25, 2))
    tkpts[1, :] = (1000, 100)
    tkpts[2, :] = (900, 100)
    tkpts[5, :] = (1100, 100)
    tkpts[3, :] = (800, 200)
    tkpts[6, :] = (1200, 200)
    tkpts[4, :] = (700, 300)
    tkpts[7, :] = (1200, 300)
    tkpts[8, :] = (1000, 250)
    tkpts[9, :] = (950, 300)
    tkpts[12, :] = (1050, 300)
    tkpts[10, :] = (950, 400)
    tkpts[13, :] = (1050, 400)
    tkpts[11, :] = (950, 500)
    tkpts[14, :] = (1050, 500)
    tkpts = tkpts.T
    
    for j in range(kpts.shape[1]):
        if j in [1, 2, 5, 3, 6, 4, 7, 8, 9, 12, 10, 13, 11, 14]:
            print(kpts[0, j], kpts[1, j])

            img = cv2.circle(img, (int(kpts[0, j]), int(kpts[1, j])), \
                        radius=1, color=(255, 0, 0), thickness=4)
            cv2.putText(img, f'{j}',(int(kpts[0, j]), int(kpts[1, j])), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0), 1, cv2.LINE_AA )

            cv2.putText(img, f'{kpts[2, j]:.2f}',(int(tkpts[0, j]), int(tkpts[1, j])), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (0, 0, 255), 2, cv2.LINE_AA )
    return img

input_dir = '/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/Color_flip_rename_openpose'
img_dir = input_dir
save_dir = '/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/Color_flip_rename_openpose_new'

os.makedirs(save_dir, exist_ok=True)
for one in tqdm(glob.glob(os.path.join(input_dir, '*.json'))):
    # keyp_tuple = read_keypoints(one)
    with open(one, 'r') as fin:
        result = json.load(fin)
        # import pdb;pdb.set_trace()
        kpts = np.array(result['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    
    # import pdb;pdb.set_trace()
    basename = os.path.basename(one)
    img_id = basename.split('_')[0]
    img_fn = os.path.join(img_dir, img_id + '_openpose.png')


    img = cv2.imread(img_fn)
    new_img = draw_op(kpts, img)
    # cv2.imshow('kpts', new_img)
    # cv2.waitkey()

    cv2.imwrite(os.path.join(save_dir, img_id+'_op.jpg'), new_img)


