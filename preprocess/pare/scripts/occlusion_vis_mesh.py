import os
import cv2
import sys
import torch
import joblib
import trimesh
import numpy as np
from tqdm import tqdm
from smplx import SMPL
import skimage.io as io
from os.path import join
import matplotlib as mpl
import neural_renderer as nr
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30
from trimesh.visual import color
from standard_rasterize_pkg.visibility import *
from matplotlib import cm as mpl_cm, colors as mpl_colors

from pare.utils.renderer import Renderer
from pare.core.config import SMPL_MODEL_DIR
from pare.utils.kp_utils import get_common_joint_names

IMG_SIZE = 480
FOCAL_L = 5000.
device = 'cuda'

PARE_DIR = 'logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train/occlusion_test_train'
SPIN_DIR = 'logs/pare_coco/05.11-spin_ckpt_eval/logs/occlusion_test_train'

def get_camera_extrinsic(cam_t):
    camera_pose = np.eye(3)
    # camera_pose[:3, 3] = cam_t
    R = torch.from_numpy(camera_pose).float()
    # print(cam_t)
    # cam_t[0] *= -1.
    cam_t[1] *= -1.
    # print(cam_t)
    t = torch.from_numpy(cam_t).float()
    return R, t


def get_camera_intrinsic():
    K = torch.Tensor([
        [FOCAL_L, 0,       IMG_SIZE/2],
        [0,       FOCAL_L, IMG_SIZE/2],
        [0,       0,       1         ],
    ]).float()
    return K


def get_texture_color(tex_f, cam_f, vertices, faces):

    rp_grph = cv2.cvtColor(cv2.imread(tex_f), cv2.COLOR_BGR2RGB)

    rp_nmr_textures = np.zeros((1, faces.shape[1], 1, 1, 1, 3), dtype=np.float32)

    R, t = get_camera_extrinsic(np.load(cam_f))
    R = R.to(device)[None, ...]
    t = t.to(device)[None, ...]
    K = get_camera_intrinsic().to(device)[None, ...]
    # P = torch.matmul(K, RT).unsqueeze(0)

    # vertices = projected_vertices(vertices, P)
    # dist_coeffs = torch.zeros(1,1,5,)
    dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]) # .repeat(1, 1)
    vertices = nr.projection(vertices, K=K, R=R, t=t, dist_coeffs=dist_coeffs, orig_size=IMG_SIZE)

    vert_vis, depth_buf, face_buf, f_vs = get_visibility(vertices, faces, h=IMG_SIZE, w=IMG_SIZE, return_all=True)
    face_buf = face_buf.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()

    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if face_buf[i,j] > -1:
                rp_nmr_textures[0, face_buf[i,j], 0, 0, 0, :] = rp_grph[i,j]/255.
    rp_nmr_textures = torch.from_numpy(rp_nmr_textures).to(device)

    return rp_nmr_textures


def get_texture_as_array(array, cam_f, vertices, faces):

    # rp_grph = array # cv2.cvtColor(cv2.imread(tex_f), cv2.COLOR_BGR2RGB)

    rp_nmr_textures = np.zeros((1, faces.shape[1], 1, 1, 1, 3), dtype=np.float32)

    R, t = get_camera_extrinsic(np.load(cam_f))
    R = R.to(device)[None, ...]
    t = t.to(device)[None, ...]
    K = get_camera_intrinsic().to(device)[None, ...]
    # P = torch.matmul(K, RT).unsqueeze(0)

    # vertices = projected_vertices(vertices, P)
    # dist_coeffs = torch.zeros(1,1,5,)
    dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]) # .repeat(1, 1)
    vertices = nr.projection(vertices, K=K, R=R, t=t, dist_coeffs=dist_coeffs, orig_size=IMG_SIZE)

    vert_vis, depth_buf, face_buf, f_vs = get_visibility(vertices, faces, h=IMG_SIZE, w=IMG_SIZE, return_all=True)
    face_buf = face_buf.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()

    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if face_buf[i,j] > -1:
                rp_nmr_textures[0, face_buf[i,j], 0, 0, 0, :] = array[i,j] # rp_grph[i,j]/255.
    rp_nmr_textures = torch.from_numpy(rp_nmr_textures).to(device)

    return rp_nmr_textures[0, :, 0, 0, 0, :].detach().cpu().numpy()


def save_textures(results_dir, debug=False):
    images_dir = os.path.join(results_dir, 'output_images')
    image_files = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith('.jpg')])

    for joint_idx in range(14):
        for idx, img_f in enumerate(image_files):
            print(f'Processing {idx}/{len(image_files)} {os.path.basename(img_f)}/{joint_idx:02d}')
            # img_f =
            tex_f = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_cross_norm.png')
            cam_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.npy')
            obj_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj')

            mesh = trimesh.load(obj_f)

            vertices = torch.from_numpy(mesh.vertices)
            faces = torch.from_numpy(mesh.faces)

            vertices = vertices[None, :, :].to(device).float()
            faces = faces[None, :, :].to(device).float()

            face_colors = get_texture_color(tex_f, cam_f, vertices, faces)

            face_colors = face_colors[0, :, 0, 0, 0, :].detach().cpu().numpy()  # [1, 13776, 1, 1, 1, 3]
            # vertex_colors = color.face_to_vertex_color(mesh, face_colors, dtype=float)
            # import IPython; IPython.embed(); exit()
            # vis_mask = vertex_colors[:, :3].sum(-1) > 0

            np.save(tex_f.replace('.png', '.npy'), face_colors)

            if debug:
                mesh.visual.face_colors = face_colors
                mesh.show()
            # if idx > 0:
            #     break


def show_final_mesh(results_dir, debug=False):
    images_dir = os.path.join(results_dir, 'output_images')
    image_files = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith('.jpg')])

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )
    s = smpl()
    verts = s.vertices.cpu().detach().numpy()[0]
    faces = smpl.faces
    vals = []
    for joint_idx in range(14):

        sum_tex = np.zeros((13776, 3))
        # max_tex = np.zeros((13776, 3))
        vis_count = np.zeros(13776)

        for idx, img_f in enumerate(image_files):
            # print(f'Processing {idx}/{len(image_files)} {os.path.basename(img_f)}/{joint_idx:02d}')
            # img_f =
            tex_f = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_cross_norm.npy')
            obj_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj')

            tex = np.load(tex_f)
            vis_mask = tex.sum(-1) > 0
            vis_count += vis_mask
            sum_tex += tex[:, :3]

            # breakpoint()
            if debug:
                print(tex.min(), tex.max())
                m = trimesh.load(obj_f)
                m.visual.face_colors = tex
                m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=tex)
                m.show()

        print('\n>>> Occlusion heatmap results for: ', get_common_joint_names()[joint_idx], '\n')
        avg_tex = sum_tex / (vis_count[..., None] + 1e-8)
        # This texture is gray scale
        avg_tex = avg_tex[:, 0]
        vals += [avg_tex.min(), avg_tex.max()]
        # import IPython; IPython.embed(); exit()
        # avg_tex = (avg_tex - avg_tex.mean()) / avg_tex.std()
        print(avg_tex.min(), avg_tex.max(), avg_tex.mean())
        # avg_tex = avg_tex / 0.2980392247436831
        avg_tex = hist_eq(avg_tex)
        # convert to a color map
        cmap = mpl_cm.get_cmap('jet')
        norm = mpl_colors.Normalize()
        avg_tex = norm(avg_tex)
        avg_tex = cmap(avg_tex)[:,:3]

        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=avg_tex)
        m.export(file_obj=f'logs/occlusion_meshes/{get_common_joint_names()[joint_idx]}.ply')
        m.show()

    print(min(vals), max(vals))


def hist_eq(texture, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    sh = texture.shape
    # get image histogram
    image_histogram, bins = np.histogram(texture.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(texture.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(sh)


def normalize_heatmaps(method_1_dir, method_2_dir, img, joint_id, im_size=480):
    # img_1 = 'result_00_35500.jpg'
    # img_2 = 'result_00_26600.jpg'
    # j_1, j_2 = 0, 0
    #
    # min_ = np.inf
    # max_ = -np.inf
    #
    # heatmaps = {}
    # hm_arr = []
    # for m_idx, m in enumerate((method_1_dir, method_2_dir)):
    #     for im_idx, im in enumerate((img_1, img_2)):
    #         pkl = os.path.join(m, 'output_images', im.replace('.jpg', '.pkl'))
    #         hm = joblib.load(pkl)['mpjpe_heatmap']
    #         if im == img_1:
    #             hm = hm[:, :, j_1]
    #         else:
    #             hm = hm[:, :, j_2]
    #
    #         hm = cv2.resize(hm, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
    #         min_ = min(min_, hm.min())
    #         max_ = max(max_, hm.max())
    #
    #         heatmaps[(m_idx,im_idx)] = hm
    #         hm_arr.append(hm)
    #
    # minmax = (min_, max_)
    #
    # heatmap = np.stack(hm_arr)
    # heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)  # normalize between [0,1]
    # cm = mpl_cm.get_cmap('jet')
    #
    # heatmap_color = []
    # for h in heatmap:
    #     heatmap_color.append(cm(h)[:,:,:3])

    # img = 'result_00_35500.jpg'
    # img_2 = 'result_00_26600.jpg'
    # j = 5
    j = joint_id # 13

    min_ = np.inf
    max_ = -np.inf

    im_arr = io.imread(os.path.join(method_1_dir, 'output_meshes', img))
    # heatmaps = {}
    hm_arr = []
    for m_idx, m in enumerate((method_1_dir, method_2_dir)):
        pkl = os.path.join(m, 'output_images', img.replace('.jpg', '.pkl'))
        # hm = joblib.load(pkl)['mpjpe_heatmap'][:,:,j]
        hm = joblib.load(pkl)['mpjpe_heatmap'].mean(-1)
        hm = cv2.copyMakeBorder(hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
        hm = cv2.resize(hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        # hm = torch.nn.functional.interpolate(
        #     torch.from_numpy(hm[None,None,...]), size=(im_size, im_size), align_corners=False, mode='bilinear',
        # ).numpy()[0,0]
        hm_arr.append(hm)

    minmax = (min_, max_)

    heatmap = np.stack(hm_arr)

    # heatmap = hist_eq(heatmap)
    heatmap_orig = heatmap.copy()
    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)  # normalize between [0,1]

    cm = mpl_cm.get_cmap('jet')

    heatmap_color = []
    for i, h in enumerate(heatmap):
        print(h.min(), h.max())
        h_c = cm(h)[:, :, :3]
        h_c = (h_c * 255).astype(np.uint8)
        io.imsave(f'logs/figures_temp/' + img.replace(".jpg", f"_{i}.png"), h_c)
        heatmap_color.append(h_c)

    heatmap_orig *= 1000.
    f = np.hstack([heatmap_orig[0], heatmap_orig[1]])
    plt.imshow(f, cmap='jet')
    clb = plt.colorbar()
    # clb.ax.set_ytitle('Joint error (mm)')
    # clb.ax.set_ylim(bottom=heatmap_orig.min(), top=heatmap_orig.max())
    # plt.show()
    return heatmap_color, heatmap_orig
    # import IPython; IPython.embed(); exit()


def cross_normalize():
    spin_dir = join(SPIN_DIR, 'output_images')
    pare_dir = join(PARE_DIR, 'output_images')

    im_size = 480

    ##### Read raw heatmaps #####
    pkl_files = sorted([join(spin_dir, x) for x in os.listdir(spin_dir) if x.endswith('.pkl')])
    spin_heatmaps = []
    pare_heatmaps = []
    for pf in pkl_files:
        print(os.path.basename(pf))
        spin_hm = joblib.load(pf)['mpjpe_heatmap']
        spin_heatmaps.append(spin_hm)

        pare_hm = joblib.load(pf.replace(spin_dir, pare_dir))['mpjpe_heatmap']
        pare_heatmaps.append(pare_hm)

    spin_heatmaps = np.stack(spin_heatmaps)
    pare_heatmaps = np.stack(pare_heatmaps)

    print('SPIN minmax', np.min(spin_heatmaps), np.max(spin_heatmaps))
    print('PARE minmax', np.min(pare_heatmaps), np.max(pare_heatmaps))

    _min = min(np.min(spin_heatmaps), np.min(pare_heatmaps))
    _max = max(np.max(spin_heatmaps), np.max(pare_heatmaps))

    import IPython; IPython.embed(); exit()

    ##### Save cross normalized heatmaps as images #####
    for images_dir in (spin_dir, pare_dir):
        # images_dir = os.path.join(results_dir, 'output_images')
        image_files = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith('.jpg')])

        for idx, img_f in enumerate(image_files):
            print(f'Processing {idx}/{len(image_files)} {os.path.basename(img_f)}')

            mean_tex_f = img_f.replace('.jpg', '_cross_norm.png')
            pkl_file = img_f.replace('.jpg', '.pkl')
            hm = joblib.load(pkl_file)['mpjpe_heatmap']
            cross_norm_mean_hm = (hm.mean(-1) - _min) / (_max - _min)
            cross_norm_mean_hm = cv2.copyMakeBorder(cross_norm_mean_hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
            cross_norm_mean_hm = cv2.resize(cross_norm_mean_hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
            cross_norm_mean_hm = (cross_norm_mean_hm * 255).astype(np.uint8)
            io.imsave(mean_tex_f, cross_norm_mean_hm)

            for joint_idx in range(14):
                j_tex_f = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_cross_norm.png')

                cross_norm_hm = (hm[:,:,joint_idx] - _min) / (_max - _min)

                cross_norm_hm = cv2.copyMakeBorder(cross_norm_hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
                cross_norm_hm = cv2.resize(cross_norm_hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

                cross_norm_hm = (cross_norm_hm * 255).astype(np.uint8)
                io.imsave(j_tex_f, cross_norm_hm)

        # import IPython; IPython.embed(); exit()

def remove_outliers(arr, max_deviations=2.):
    mean = np.mean(arr)
    standard_deviation = np.std(arr)
    distance_from_mean = abs(arr - mean)
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = arr[not_outlier]
    return np.clip(arr, no_outliers.min(), no_outliers.max())


def analyze_face_errors(debug=False):
    spin_dir = join(SPIN_DIR, 'output_images')
    pare_dir = join(PARE_DIR, 'output_images')
    vals = []
    avg_textures = []
    max_textures = []

    smpl = SMPL(SMPL_MODEL_DIR)
    s = smpl()
    verts = s.vertices.cpu().detach().numpy()[0]
    faces = smpl.faces

    _min, _max = -0.07644983304583111, 0.5745493712190755
    for images_dir in (spin_dir, pare_dir):
        print(images_dir)
        # images_dir = os.path.join(results_dir, 'output_images')
        image_files = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith('.jpg')])
        image_files = image_files# [::10]


        for joint_idx in range(14):

            sum_tex = np.zeros((13776, 3))
            max_tex = np.zeros((13776, 3))
            vis_count = np.zeros(13776)

            for idx, img_f in enumerate(image_files):
                # print(f'Processing {idx}/{len(image_files)} {os.path.basename(img_f)}/{joint_idx:02d}')
                # img_f =
                tex_f = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_face_errors.npy')
                obj_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj')

                tex = np.load(tex_f)
                vis_mask = tex.sum(-1) != 0
                vis_count += vis_mask
                sum_tex += tex[:, :3]

                max_tex = np.maximum(max_tex, tex)
                # breakpoint()
                if debug:
                    print(tex.min(), tex.max())
                    m = trimesh.load(obj_f)
                    m.visual.face_colors = tex
                    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=tex)
                    m.show()

            print('\n>>> Occlusion heatmap results for: ', get_common_joint_names()[joint_idx])
            avg_tex = sum_tex / (vis_count[..., None] + 1e-8)
            max_tex = max_tex[:, 0]
            # This texture is gray scale
            avg_tex = avg_tex[:, 0]
            vals += [avg_tex.min(), avg_tex.max()]
            avg_textures.append(avg_tex)
            max_textures.append(max_tex)
            # import IPython; IPython.embed(); exit()
            # avg_tex = (avg_tex - _min) / (_max-_min)
            print(avg_tex.min(), avg_tex.max(), avg_tex.mean())
            # avg_tex = avg_tex / 0.2980392247436831
            # avg_tex = hist_eq(avg_tex)
            # convert to a color map

            # avg_tex = remove_outliers(avg_tex, max_deviations=2.5)
            # plt.figure(figsize=(10,4))
            # plt.hist(avg_tex, bins=np.linspace(avg_tex.min(),avg_tex.max(),100))
            # plt.title("histogram")
            # plt.show()
            # import IPython; IPython.embed()

            cmap = mpl_cm.get_cmap('jet')
            norm = mpl_colors.Normalize()
            avg_tex = norm(avg_tex)
            avg_tex = cmap(avg_tex)[:, :3]

            # m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=avg_tex)
            # m.export(file_obj=f'logs/occlusion_meshes/{get_common_joint_names()[joint_idx]}.ply')
            # m.show()

    # -0.07644983304583111 0.5745493712190755
    print(min(vals), max(vals))
    # breakpoint()
    # concatenate
    # avg_textures_joint = avg_textures.copy()

    # avg_textures = np.concatenate(avg_textures)
    #
    # avg_textures = hist_eq(avg_textures)
    # cmap = mpl_cm.get_cmap('jet')
    # norm = mpl_colors.Normalize()
    # avg_tex_norm = norm(avg_textures)
    # avg_tex_norm = cmap(avg_tex_norm)[:, :3]
    #
    # avg_textures = avg_tex_norm.reshape(2, -1, 3)
    #
    # for idx, avg_t in enumerate(avg_textures):
    #     print(f'\n>>> {"PARE" if idx == 1 else "SPIN"} ')
    #     print(avg_t.mean())
    #     print(avg_t[avg_t != (0, 0, 0)].mean())
    #
    #     m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=avg_t)
    #     # m.export(file_obj=f'logs/occlusion_meshes/{get_common_joint_names()[joint_idx]}.ply')
    #     m.show()
    print('Saving average texture')
    joblib.dump(avg_textures, 'logs/figures/occlusion_meshes/avg_texture.pkl')
    joblib.dump(max_textures, 'logs/figures/occlusion_meshes/max_texture.pkl')


    # avg_tex = np.concatenate(avg_textures)
    # # avg_tex = np.concatenate([avg_textures[0], avg_textures[14]])
    # print(avg_tex[:avg_tex.shape[0] // 2].mean())
    # print(avg_tex[avg_tex.shape[0] // 2:].mean())
    # print('normalize texture')
    # # normalize
    # avg_tex = hist_eq(avg_tex)
    # cmap = mpl_cm.get_cmap('jet')
    # norm = mpl_colors.Normalize()
    # avg_tex_norm = norm(avg_tex)
    # avg_tex_norm = cmap(avg_tex_norm)[:, :3]
    #
    # # reshape
    # avg_tex = avg_tex_norm.reshape(2, -1, 3)
    #
    # for idx, avg_t in enumerate(avg_tex):
    #     print(f'\n>>> {"PARE" if idx > 13 else "SPIN"} '
    #           f'Occlusion heatmap results for: ', get_common_joint_names()[idx % 14])
    #     print(avg_t.mean())
    #     print(avg_t[avg_t != (0,0,0)].mean())
    #
    #     m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=avg_t)
    #     # m.export(file_obj=f'logs/occlusion_meshes/{get_common_joint_names()[joint_idx]}.ply')
    #     m.show()


def plot_face_errors(debug=False):
    im_res = 1080
    smpl = SMPL(SMPL_MODEL_DIR)
    s = smpl()
    verts = s.vertices.cpu().detach().numpy()[0]
    faces = smpl.faces
    cam_t = np.array([0, 0, 2 * 5000 / im_res])
    # renderer = Renderer(focal_length=5000, img_res=im_res, faces=faces)
    avg_textures_joint = joblib.load('logs/figures/occlusion_meshes/avg_texture.pkl')

    print('Showing per joint resultsssss')
    for joint_idx in range(15):
        if joint_idx == 14:
            spin_tex, pare_tex = avg_textures_joint[joint_idx], avg_textures_joint[14:]
            avg_tex = np.stack(avg_textures_joint)
            avg_tex = avg_tex.reshape((2,-1,13776))
            avg_tex = avg_tex.mean(1)
            avg_tex = np.concatenate([avg_tex[0], avg_tex[1]])
            # breakpoint()
            jname = 'all'
        else:
            spin_tex, pare_tex = avg_textures_joint[joint_idx], avg_textures_joint[13 + joint_idx]
            avg_tex = np.concatenate([spin_tex, pare_tex])
            jname = get_common_joint_names()[joint_idx]

        avg_tex = remove_outliers(avg_tex, max_deviations=2.5)

        if debug:
            plt.figure(figsize=(10, 4))
            plt.rcParams['font.size'] = 12
            bins = np.linspace(avg_tex.min(), avg_tex.max(), 100)
            plt.hist(avg_tex[:avg_tex.shape[0]//2], bins=bins, label='spin')
            plt.hist(avg_tex[avg_tex.shape[0]//2:], bins=bins, label='pare')
            plt.legend()
            plt.title("histogram")
            plt.show()

        orig_avg_tex = avg_tex.copy() * 1000

        # avg_tex = hist_eq(avg_tex)
        cmap = mpl_cm.get_cmap('jet')
        norm = mpl_colors.Normalize()
        avg_tex_norm = norm(avg_tex)
        avg_tex_norm = cmap(avg_tex_norm)[:, :3]

        avg_tex = avg_tex_norm.reshape(2, -1, 3)


        for idx, avg_t in enumerate(avg_tex):
            print(f'\n>>> {"PARE" if idx == 1 else "SPIN"} '
                  f'Occlusion heatmap results for: ', jname)
            print(avg_t[avg_t != (0, 0, 0)].mean())

            print('saving!...')
            method = 'pare' if idx == 1 else 'spin'
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False, face_colors=avg_t)
            # m.show()

            m.export(file_obj=f'logs/figures/occlusion_meshes/{method}_{jname}.ply')

            # if debug:
            #     m.show()

            scene = m.scene()

            rend_mesh = scene.save_image(resolution=[im_res, im_res])


            filename_f = f'logs/figures/occlusion_meshes/{method}_{jname}_front.png'
            with open(filename_f, 'wb') as f:
                f.write(rend_mesh)
                f.close()

            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [0, 1, 0])
            m.apply_transform(rot)

            rend_mesh = scene.save_image(resolution=[im_res, im_res])
            filename_b = f'logs/figures/occlusion_meshes/{method}_{jname}_back.png'
            with open(filename_b, 'wb') as f:
                f.write(rend_mesh)
                f.close()

            # io.imsave(f'logs/figures/occlusion_meshes/{method}_{get_common_joint_names()[joint_idx]}.png', rend_mesh)

            plt.rcParams['font.size'] = 150
            # plt.rcParams["font.weight"] = "bold"
            fig, ax = plt.subplots(figsize=(2, 35))
            cmap = mpl.cm.jet
            norm = mpl.colors.Normalize(vmin=orig_avg_tex.min(), vmax=orig_avg_tex.max())
            cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
            # cb.set_label('Mean joint error (mm)')

            fig.savefig(f'logs/figures/occlusion_meshes/{method}_{jname}_colorbar.png',
                        bbox_inches='tight', transparent=True)
            plt.close('all')

            # exit()

            if debug:
                cv2.imshow('img', cv2.cvtColor(filename_f, cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow('img', cv2.cvtColor(filename_b, cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def save_face_errors(debug=False):
    spin_dir = join(SPIN_DIR, 'output_images')
    pare_dir = join(PARE_DIR, 'output_images')

    im_size = 480

    ##### Read raw heatmaps #####
    pkl_files = sorted([join(spin_dir, x) for x in os.listdir(spin_dir) if x.endswith('.pkl')])
    spin_heatmaps = []
    pare_heatmaps = []
    for pf in pkl_files:
        print(os.path.basename(pf))
        spin_hm = joblib.load(pf)['mpjpe_heatmap']
        spin_heatmaps.append(spin_hm)

        pare_hm = joblib.load(pf.replace(spin_dir, pare_dir))['mpjpe_heatmap']
        pare_heatmaps.append(pare_hm)

    spin_heatmaps = np.stack(spin_heatmaps)
    pare_heatmaps = np.stack(pare_heatmaps)

    print('SPIN minmax', np.min(spin_heatmaps), np.max(spin_heatmaps))
    print('PARE minmax', np.min(pare_heatmaps), np.max(pare_heatmaps))

    _min = min(np.min(spin_heatmaps), np.min(pare_heatmaps))
    _max = max(np.max(spin_heatmaps), np.max(pare_heatmaps))

    # import IPython; IPython.embed(); exit()

    ##### Save cross normalized heatmaps as images #####
    for images_dir in (spin_dir, pare_dir):
        # images_dir = os.path.join(results_dir, 'output_images')
        image_files = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith('.jpg')])
        image_files = image_files
        for idx, img_f in enumerate(image_files):
            print(f'Processing {idx}/{len(image_files)} {os.path.basename(img_f)}')

            # mean_tex_f = img_f.replace('.jpg', '_cross_norm.png')
            pkl_file = img_f.replace('.jpg', '.pkl')
            hm = joblib.load(pkl_file)['mpjpe_heatmap']

            hm = cv2.copyMakeBorder(hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
            hm = cv2.resize(hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

            cam_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.npy')
            obj_f = img_f.replace('output_images', 'output_meshes').replace('.jpg', '.obj')
            mesh = trimesh.load(obj_f)
            vertices = torch.from_numpy(mesh.vertices)
            faces = torch.from_numpy(mesh.faces)
            vertices = vertices[None, :, :].to(device).float()
            faces = faces[None, :, :].to(device).float()
            face_errors = get_texture_as_array(hm.mean(-1), cam_f, vertices, faces)

            if debug:
                mesh.visual.face_colors = face_errors
                mesh.show()

            np.save(pkl_file.replace('.pkl', '_face_errors.npy'), face_errors)


            for joint_idx in range(14):
                # j_tex_f = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_cross_norm.png')

                joint_hm = hm[:,:,joint_idx]

                joint_hm = cv2.copyMakeBorder(joint_hm, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
                joint_hm = cv2.resize(joint_hm, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
                joint_face_errors = get_texture_as_array(joint_hm, cam_f, vertices, faces)
                j_save_file = os.path.join(img_f.replace('.jpg', '_hm'), f'{joint_idx:02d}_face_errors.npy')
                np.save(j_save_file, joint_face_errors)

            # breakpoint()
        # import IPython; IPython.embed(); exit()


if __name__ == '__main__':
    results_dir = PARE_DIR

    if sys.argv[1] == 'save':
        save_textures(results_dir, debug=False)
    elif sys.argv[1] == 'show':
        show_final_mesh(results_dir, debug=False)
    elif sys.argv[1] == 'norm':
        normalize_heatmaps(PARE_DIR, SPIN_DIR, img=sys.argv[2], joint_id=int(sys.argv[3]))
    elif sys.argv[1] == 'cross':
        cross_normalize()
    elif sys.argv[1] == 'face':
        save_face_errors(debug=False)
    elif sys.argv[1] == 'analyze':
        # analyze_face_errors(debug=False)
        plot_face_errors(debug=False)


