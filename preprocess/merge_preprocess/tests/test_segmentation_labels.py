import torch
import numpy as np
import neural_renderer as nr
import skimage.io as io
import skimage.transform as tr
import matplotlib.pyplot as plt

from pare.core.config import SMPL_MODEL_DIR
from pare.models.head.smpl_head import SMPL
from pare.utils.image_utils import get_body_part_texture, generate_part_labels, get_default_camera
from pare.utils.vis_utils import visualize_segm_masks


def main():
    device = 'cuda'

    imsize = 1080
    batch_size = 3
    smpl = SMPL(SMPL_MODEL_DIR).to(device)

    neural_renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=imsize,
        image_size=imsize,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=False,
    )

    cam = torch.zeros(batch_size, 3, device=device)
    cam[:, 0] = 1.0

    cam_t = torch.stack(
        [
            cam[:, 1],
            cam[:, 2],
            2 * 5000. / (imsize * cam[:, 0] + 1e-9)
        ],
        dim=-1
    )
    n_parts = 24

    K, R = get_default_camera(5000., imsize)
    K, R = K.to(device), R.to(device)
    smpl_faces = torch.from_numpy(smpl.faces.astype(np.int32)).unsqueeze(0).to(device)
    body_part_texture = get_body_part_texture(smpl.faces, n_vertices=6890).to(device)
    bins = (torch.arange(int(n_parts)) / float(n_parts) * 255.) + 1
    bins = bins.to(device)
    global_orient = torch.zeros(batch_size, 3, device=device)
    global_orient[:, 0] = np.pi
    body_pose = torch.zeros(batch_size, 69, device=device) # torch.rand(batch_size, 69, device=device) * 0.4 - 0.2

    vertices = smpl(
        body_pose=body_pose,
        global_orient=global_orient
    ).vertices


    parts, render = generate_part_labels(
        vertices=vertices,
        faces=smpl_faces,
        cam_t=cam_t,
        K=K,
        R=R,
        body_part_texture=body_part_texture,
        neural_renderer=neural_renderer,
        part_bins=bins,
    )
    print(parts.shape)
    print(torch.unique(parts))

    # render = render.detach().cpu().numpy()[0].transpose(1,2,0)

    parts = parts.cpu().numpy()

    # import IPython; IPython.embed(); exit()

    for i in range(batch_size):
        image = np.zeros((imsize,imsize,3)) # io.imread('../00000.jpg') # '/home/mkocabas/Pictures/00000.jpg')
        image = tr.resize(image, output_shape=(imsize, imsize))
        alpha = 1.0
        mask = parts[i]

        mask_ch = np.zeros((n_parts+1, mask.shape[0], mask.shape[1]))
        for i in range(n_parts+1):
            print(i)
            indices = np.argwhere(mask == i)
            mask_ch[i, indices[:,0], indices[:,1]] = 1.

        mask = mask_ch
        # dummmy_mask
        # mask = np.expand_dims(parts[i], axis=0).repeat(n_parts+1, axis=0)
        # mask = np.zeros_like(mask)
        # for j in range(mask.shape[0]):
        #     mask[j, j*7:j*7 + 7, j*7:j*7 + 7] = 1.

        mask_img = visualize_segm_masks(image, mask=mask, alpha=alpha)
        # plt.imshow(mask)
        plt.imshow(mask_img, cmap='gist_ncar')
        # plt.savefig(f'test_{i}.png')
        mask_img[mask_img == 0] = 1.0
        io.imsave('data/body_part_segm.png', mask_img)
        plt.show()



if __name__ == '__main__':
    main()
#
# 1.  Hips
# 2.  Left Hip
# 3.  Right Hip
# 4.  Spine
# 5.  Left Knee
# 6.  Right Knee
# 7.  Spine_1
# 8.  Left Ankle
# 9.  Right Ankle
# 10. Spine_2
# 11. Left Toe
# 12. Right Toe
# 13. Neck
# 14. Left Shoulder
# 15. Right Shoulder
# 16. Head
# 17. Left Arm
# 18. Right Arm
# 19. Left Elbow
# 20. Right Elbow
# 21. Left Hand
# 22. Right Hand
# 23. Left Thumb
# 24. Right Thumb