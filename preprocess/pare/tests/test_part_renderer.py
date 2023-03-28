import torch
import numpy as np
import neural_renderer as nr
import matplotlib.pyplot as plt
# import config
import joblib
from trimesh.visual import color

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

from smplx.body_models import SMPL
from pare.core.config import SMPL_MODEL_DIR

def get_parts(parts, mask, cube_parts):
    """Process renderer part image to get body part indices."""
    bn,c,h,w = parts.shape
    mask = mask.view(-1,1)
    parts_index = torch.floor(100*parts.permute(0,2,3,1).reshape(-1,3)).long()
    parts = cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None]
    parts *= mask
    parts = parts.view(bn,h,w).long()
    return parts

def main():
    device = 'cuda'

    # textures = torch.from_numpy(np.load('data/vertex_texture.npy')).to(device).float()
    # cube_parts = torch.cuda.FloatTensor(np.load('data/cube_parts.npy'))
    smpl = SMPL(SMPL_MODEL_DIR).to(device)
    faces = torch.from_numpy(smpl.faces.astype(np.int32)).to(device)

    textures = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)

    focal_length = 5000.
    img_size = 224

    renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=img_size,
        image_size=img_size,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=False
    )

    betas = torch.zeros(1, 10, device=device)
    body_pose = torch.rand(1, 69, device=device) * 0.4 - 0.2 # torch.zeros(1, 69, device=device)
    global_orient = torch.zeros(1, 3, device=device)
    global_orient[:,0] = np.pi

    vertices = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
    ).vertices

    # import trimesh
    # m = trimesh.load('/is/cluster/work/mkocabas/projects/pare/logs/demo_results/friends_bradpitt_0.25-0.37_spin/meshes/0001/000010.obj')
    # vertices = m.vertices
    # vertices = torch.from_numpy(vertices).float().to(device).unsqueeze(0)

    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')
    # vertex_colors = np.random.random(size=[vertices.shape[0], 4])

    vertex_colors = np.ones((vertices.shape[1], 4)) * np.array([0.3, 0.3, 0.3, 1.])
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    # norm_gt = mpl_colors.Normalize(vmin=0, vmax=trunc_val)
    # vertex_colors[:, :3] = cm(norm_gt(smpl_segmentation['smpl_index']))[:, :3]
    vertex_colors[:, :3] = smpl_segmentation['smpl_index'][..., None]
    # face_colors = color.vertex_to_face_color(vertex_colors, faces.cpu().numpy())
    vertex_colors = color.to_rgba(vertex_colors)
    face_colors = vertex_colors[faces.cpu().numpy()].min(axis=1)
    # breakpoint()
    textures[0, :, 0, 0, 0, :] = face_colors[:,:3] / 24.
    textures = torch.from_numpy(textures).float().to(device)

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
    parts, depth, mask = renderer(
        vertices,
        faces,
        textures=textures.expand(batch_size, -1, -1, -1, -1, -1),
        K=K,
        R=R,
        t=cam_t.unsqueeze(1)
    )

    # import IPython; IPython.embed(); exit(1)
    print(textures.shape)
    # parts = get_parts(parts, mask, cube_parts)
    parts = parts.permute(0,2,3,1).detach().cpu().numpy()
    parts *= 255.
    parts = parts.max(-1)

    bins = (np.arange(24) / 24. * 255.) + 1
    parts = np.digitize(parts, bins, right=True)
    parts = parts.astype(np.uint8) + 1
    print(parts.shape)
    print(np.unique(parts))
    plt.imshow(parts[0])
    plt.show()
    # plt.savefig('parts.png')
    # import IPython; IPython.embed(); exit(1)




if __name__ == '__main__':
    main()