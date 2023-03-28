
import cv2
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d import io
import neural_renderer as nr

from standard_rasterize_pkg.visibility import *

device = 'cuda' if torch.cuda.is_available() else sys.exit('GPU not available!!')

IMG_SIZE = 1000
FOCAL_L = 1000.


def show_imgs(imgs, size=15):
    num_imgs = len(imgs)
    fig, axs = plt.subplots(1, num_imgs, squeeze=False,
                            figsize=(size,size))
    for i in range(num_imgs):
        axs[0,i].imshow(imgs[i])
        axs[0,i].axis('off')
    plt.show()


def preprocess_smplx(smplxfile, scanfile):
    vertices, faces_idx, rp_aux = io.load_obj(smplxfile, load_textures=False)
    faces = faces_idx.verts_idx

    rp_scan_vert, _, _ = io.load_obj(scanfile)
    rp_scan_vert = rp_scan_vert/100
    rp_scan_vert = torch.mean(rp_scan_vert, 0)
    vertices -= rp_scan_vert

    vertices = vertices[None, :, :].to(device)
    faces = faces[None, :, :].to(device)

    return vertices, faces


# def projected_vertices(vertices, P):
#     dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)
#     vertices = nr.projection(vertices, P, dist_coeffs, IMG_SIZE)
#     return vertices


def get_texture_color(grphfile, vertices, faces):

    rp_grph = cv2.cvtColor(cv2.imread(grphfile), cv2.COLOR_BGR2RGB)

    rp_nmr_textures = np.zeros((1, faces.shape[1], 1, 1, 1, 3), dtype=np.float32)

    # vertices = projected_vertices(vertices, P)

    vert_vis, depth_buf, face_buf, f_vs = get_visibility(
        vertices, faces, h=IMG_SIZE, w=IMG_SIZE, return_all=True
    )

    face_buf = face_buf.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()

    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if face_buf[i,j] > -1:
                rp_nmr_textures[0, face_buf[i,j], 0, 0, 0, :] = rp_grph[i,j] / 255.
    rp_nmr_textures = torch.from_numpy(rp_nmr_textures).to(device)

    return rp_nmr_textures


def Renderer(vertices, faces, tex_color, projection_mat):

    projection_mat[0,1,1] *= -1.0
    renderer = nr.Renderer(
        camera_mode='projection',
        K=None,
        R=None,
        t=None,
        orig_size=IMG_SIZE,
        image_size=IMG_SIZE,
        camera_direction=[1,1,1],
        light_direction=[0,0,-1]
    )
    images = renderer(vertices, faces, tex_color)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    rp_render_nmr = (image * 255).astype(np.uint8)

    return rp_render_nmr


if __name__ == '__main__':
    pass