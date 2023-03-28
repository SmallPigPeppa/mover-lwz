import torch
import numpy as np
import neural_renderer as nr
import matplotlib.pyplot as plt

from smplx.body_models import SMPL

def test():
    item_list = [5, 4, 4, 6, 7]

    for i in range(len(item_list)):
        idx = len(item_list) - i
        print(item_list[idx])


def main():
    device = 'cuda'

    textures = torch.from_numpy(np.load('vertex_texture.npy')).to(device).float()
    smpl = SMPL('smpl').to(device)
    faces = torch.from_numpy(smpl.faces.astype(np.int32)).to(device)

    focal_length = 5000.
    img_size = 224

    renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=img_size,
        image_size=img_size,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=True
    )

    betas = torch.zeros(1, 10, device=device)
    body_pose = torch.zeros(1, 69, device=device)
    global_orient = torch.zeros(1, 3, device=device)
    global_orient[:,0] = np.pi

    vertices = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
    ).vertices

    cam = torch.zeros(1,3, device=device)
    cam[:, 0] = 1.0

    cam_t = torch.stack(
        [
            cam[:, 1],
            cam[:, 2],
            2 * focal_length / (img_size * cam[:, 0] + 1e-9)
        ],
        dim=-1
    )

    batch_size = vertices.shape[0]
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1
    K[0, 2] = img_size / 2.
    K[1, 2] = img_size / 2.
    K = K[None, :, :].expand(batch_size, -1, -1)
    R = torch.eye(3, device=device)[None, :, :].expand(batch_size, -1, -1)

    faces = faces[None, :, :].expand(batch_size, -1, -1)
    parts, _, mask = renderer(
        vertices,
        faces,
        textures=textures.expand(batch_size, -1, -1, -1, -1, -1),
        K=K,
        R=R,
        t=cam_t.unsqueeze(1)
    )

    # import IPython; IPython.embed(); exit(1)

    plt.imshow(parts[0].detach().cpu().numpy().transpose(1,2,0))
    plt.savefig('parts.png')
    # import IPython; IPython.embed(); exit(1)


if __name__ == '__main__':
    main()