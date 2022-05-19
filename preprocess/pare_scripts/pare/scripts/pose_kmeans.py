import cv2
import math
import torch
import argparse
import numpy as np
from smplx import SMPL
from skimage.io import imsave
from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import silhouette_samples, silhouette_score
from human_body_prior.tools.model_loader import load_vposer

from pare.utils.renderer import Renderer
from pare.core.config import DATASET_FILES, SMPL_MODEL_DIR


def main(args):
    use_vposer = args.vposer
    dataset = np.load(DATASET_FILES[0]['3dpw-all'])
    poses = dataset['pose']

    vp, ps = load_vposer('data/vposer_v1_0')

    if use_vposer:
        print('Using VPoser to reduce dimensionality')

        poses = poses[:,3:-6]

        vp_inp = torch.from_numpy(poses).float()
        vp_out = vp.encode(vp_inp).rsample()
        poses = vp_out.detach().cpu().numpy()
    else:
        poses = poses[:, 3:]

    print('Pose dataset size', poses.shape)

    n_cluster = args.n_cluster
    print(f'Clustering into {n_cluster} clusters')

    kmeans = KMeans(
        n_clusters=n_cluster,
        algorithm='full', # “auto”, “full”, “elkan”
        init='k-means++',
        # random_state=0
    ).fit(poses)

    tsne = TSNE(n_components=2)

    canonical_poses = kmeans.cluster_centers_
    labels = kmeans.labels_

    silhouette_avg = silhouette_score(poses, labels)

    print('Silhoutte score: ', silhouette_avg)

    if use_vposer:
        vp_inp = torch.from_numpy(canonical_poses).float()
        vp_out = vp.decode(vp_inp, output_type='aa')
        canonical_poses = vp_out.reshape(n_cluster, -1).detach().cpu().numpy()

    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )
    renderer = Renderer(faces=smpl.faces)
    canonical_poses = torch.from_numpy(canonical_poses).float()

    global_orientation = torch.zeros(n_cluster, 3)
    global_orientation[:, 0] = np.pi
    canonical_poses = torch.cat([global_orientation, canonical_poses], dim=-1)

    if use_vposer:
        hand_rot = torch.zeros(n_cluster, 6)
        canonical_poses = torch.cat([canonical_poses, hand_rot], dim=-1)

    vertices = smpl(
        betas=torch.zeros(n_cluster, 10),
        body_pose=canonical_poses[:, 3:],
        global_orient=canonical_poses[:, :3]
    ).vertices

    cam_t = np.array([0, 0, 2*5000/224])

    images = []

    print('Rendering cluster centers..')
    for idx in range(vertices.shape[0]):
        verts = vertices[idx].cpu().numpy()

        image = renderer(
            vertices=verts,
            camera_translation=cam_t,
            image=np.ones((224,224,3)),
            sideview=False,
        )

        image = cv2.putText(image, f'{idx:02d}', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 0, 0))

        percentage = (labels[labels == idx].shape[0] / labels.shape[0]) * 100

        image = cv2.putText(image, f'{percentage:.2f}%', (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 0, 0))

        images.append(image)

    grid_image = make_grid(torch.Tensor(np.array(images).transpose((0,3,1,2))), nrow=int(math.sqrt(n_cluster)))

    fname = f'data/clustering/vposer_n{n_cluster}_3dpw_clusters.png' \
        if use_vposer else f'data/clustering/rawpose_n{n_cluster}_3dpw_clusters.png'

    imsave(fname, grid_image.numpy().transpose(1,2,0))# np.hstack(images))

    # import IPython; IPython.embed()

    print('Running TSNE...')

    # if use_global_orientation:
    #     canonical_poses = canonical_poses.numpy()
    # else:
    #     canonical_poses = canonical_poses[:, 3:].numpy()
    #
    # poses = np.vstack([poses, canonical_poses])
    # labels = np.concatenate([labels, np.arange(n_cluster, n_cluster*2)])

    Y = tsne.fit_transform(poses)

    plt.title(f'T-SNE plots of poses using {"vposer" if use_vposer else "raw-pose"} embeddings')
    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, label=np.arange(n_cluster), cmap='jet')

    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)

    # plt.legend()
    # plt.axis('off')
    fname = f'data/clustering/vposer_n{n_cluster}_3dpw_tsne.png' \
        if use_vposer else f'data/clustering/rawpose_n{n_cluster}_3dpw_tsne.png'
    plt.savefig(fname)

    fname = f'data/clustering/vposer_n{n_cluster}_3dpw_labels.npy' \
        if use_vposer else f'data/clustering/rawpose_n{n_cluster}_3dpw_labels.npy'

    print(f'Saving cluster labels {fname}...')
    np.save(fname, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_cluster', type=int, default=10)
    parser.add_argument('--vposer', action='store_true')

    args = parser.parse_args()

    main(args)