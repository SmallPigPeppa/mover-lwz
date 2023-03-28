import numpy as np

def mview_fits():
    orig_mpii3d = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')
    mview_fits = np.load('data/static_fits/mpi-inf-3dhp_mview_fits.npz')

    print(f'saving mpi_inf_3dhp_mview_train.npz...')

    np.savez(
        'data/dataset_extras/mpi_inf_3dhp_mview_train.npz',
        imgname=orig_mpii3d['imgname'],
        center=orig_mpii3d['center'],
        scale=orig_mpii3d['scale'],
        part=orig_mpii3d['part'],
        S=orig_mpii3d['S'],
        pose=mview_fits['pose'],
        shape=mview_fits['shape'],
        has_smpl=mview_fits['has_smpl'],
        openpose=orig_mpii3d['openpose'],
    )

def spin_fits():
    orig_mpii3d = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')
    spin_fits = np.load('data/static_fits/mpi-inf-3dhp_fits.npy')

    print(f'saving mpi_inf_3dhp_spin_train.npz...')

    np.savez(
        'data/dataset_extras/mpi_inf_3dhp_spin_train.npz',
        imgname=orig_mpii3d['imgname'],
        center=orig_mpii3d['center'],
        scale=orig_mpii3d['scale'],
        part=orig_mpii3d['part'],
        S=orig_mpii3d['S'],
        pose=spin_fits[:, :72],
        shape=spin_fits[:, 72:],
        has_smpl=np.ones(spin_fits.shape[0]),
        openpose=orig_mpii3d['openpose'],
    )

if __name__ == '__main__':
    spin_fits()