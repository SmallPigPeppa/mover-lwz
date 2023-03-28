import cv2
import torch
import numpy as np
# from smplx import SMPL
import matplotlib.pyplot as plt

from pare.models.head.smpl_head import SMPL
from pare.core.config import SMPL_MODEL_DIR
from pare.utils.vis_utils import draw_skeleton


def main():
    res = 480
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        global_orient=torch.from_numpy(np.array([[np.pi, 0, 0]])).float()
    )
    joints = smpl().joints
    joints2d = joints[:, :, :2]
    joints2d -= (joints2d[:, 27, :] + joints2d[:, 28, :]) / 2
    joints2d = torch.cat([joints2d, torch.ones(1, joints2d.shape[1], 1)], dim=-1)
    joints2d = joints2d[0, 25:39, :].detach().numpy()

    image = np.ones((res, res, 3)) * 255
    image = draw_skeleton(image, kp_2d=joints2d, res=res)

    # plt.imshow(image)
    # plt.show()

    image_points = 0.5 * res * (joints2d[:, :2] + 1).astype(np.float32).reshape(1,-1,2)
    object_points = joints[0, 25:39].detach().numpy().astype(np.float32).reshape(1,-1,3)

    cv2
    # import IPython; IPython.embed(); exit()
    camera_matrix = np.eye(3)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (res, res),
        cameraMatrix=None,
        distCoeffs=None,
        # flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    print('ret', ret)
    print('mtx', mtx)
    print('dist', dist)
    print('rvecs', rvecs)
    print('tvecs', tvecs)

if __name__ == '__main__':
    main()