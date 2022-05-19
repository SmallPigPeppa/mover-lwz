import torch
import numpy as np
import skimage.io as io
import skimage.transform as tr
import matplotlib.pyplot as plt

from pare.utils.image_utils import generate_heatmaps_2d
from pare.utils.vis_utils import visualize_heatmaps
from pare.models.head.pare_head import PareHead

def get_max_preds_torch(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    # idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals


def get_max_preds_np(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


if __name__ == '__main__':

    heatmaps, joint_vis = generate_heatmaps_2d(
        joints=np.random.random((24,2))-0.5,
        joints_vis=np.ones((24,1)),
        num_joints=24,
        heatmap_size=56,
        image_size=224,
        sigma=2,
    )

    preds, max_vals = get_max_preds_np(batch_heatmaps=heatmaps[None, ...])
    print('numpy: ', preds[0, :4])#, max_vals[0])

    preds, max_vals = get_max_preds_torch(batch_heatmaps=torch.from_numpy(heatmaps[None, ...]))
    print('torch: ', preds[0, :4])#, max_vals[0])

    head = PareHead(24, 2048, softmax_temp=1.0)
    keypoints, _ = head._softargmax2d(heatmaps=torch.from_numpy(heatmaps[None, ...]), normalize_keypoints=False)
    print('softargmax: ', preds[0, :4])

    image = io.imread('/home/mkocabas/Pictures/00000.jpg')
    image = tr.resize(image, output_shape=(480,480))
    alpha = 0.3
    hm_img = visualize_heatmaps(image, heatmaps)

    plt.imshow(hm_img)
    plt.show()
