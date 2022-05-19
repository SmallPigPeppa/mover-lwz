import os
import cv2
import numpy as np
from tqdm import tqdm

from pare.core.config import LSPET_ROOT

if __name__ == '__main__':
    image_fns = [os.path.join(LSPET_ROOT, x) for x in os.listdir(LSPET_ROOT)]

    lspet_npz = LSPET_ROOT + '_npz'
    os.makedirs(lspet_npz, exist_ok=True)

    for image_fn in tqdm(image_fns[5000:]):
        try:
            cv_img = cv2.imread(image_fn)[:,:,::-1].copy().astype(np.float32)
            np.savez(image_fn.replace(LSPET_ROOT,lspet_npz), cv_img)
        except:
            print(image_fn)