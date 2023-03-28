import os
import numpy as np
from tqdm import tqdm

from pare.core.config import DATASET_FILES, DATASET_NPZ_PATH


pw3d_all = np.load(DATASET_FILES[0]['3dpw-all'])
pw3d = np.load(DATASET_FILES[0]['3dpw'])

imgname_pw3d_all = pw3d_all['imgname'].tolist()

indices = []

for pw3d_idx, img in tqdm(enumerate(pw3d['imgname'])):
    c = pw3d['center'][pw3d_idx]
    idxs = [j for j in range(len(imgname_pw3d_all))
            if img == imgname_pw3d_all[j] and np.array_equal(pw3d_all['center'][j], c)]
    assert len(idxs) == 1, 'there are more than 1 index'
    indices.append(idxs[0])

out_file = os.path.join(DATASET_NPZ_PATH, '3dpw_test_with_mmpose.npz')

print(f'saving {out_file}...')
np.savez(
        out_file,
        imgname=pw3d['imgname'],
        center=pw3d['center'],
        scale=pw3d['scale'],
        pose=pw3d['pose'],
        shape=pw3d['shape'],
        gender=pw3d['gender'],
        mmpose_keypoints=pw3d_all['mmpose_keypoints'][indices],
        openpose=pw3d_all['openpose'][indices],
    )
