import sys
sys.path.append('.')
from pare.core import config as cfg
from pare.dataset.preprocess.pw3d import pw3d_extract
from pare.dataset.preprocess.aich import aich_extract
from pare.dataset.preprocess.mpii_test import mpii_extract
from pare.dataset.preprocess.coco_test import coco_extract
from pare.dataset.preprocess.ochuman import ochuman_extract
from pare.dataset.preprocess.muco3dhp import muco3dhp_extract
from pare.dataset.preprocess.crowdpose import crowdpose_extract
from pare.dataset.preprocess.oh3d_test import oh3d_test_extract
from pare.dataset.preprocess.convert_eft_data import eft_extract
from pare.dataset.preprocess.oh3d_train import oh3d_train_extract
from pare.dataset.preprocess.mpi_inf_3dhp import mpi_inf_3dhp_extract


if __name__ == '__main__':
    datasets = sys.argv[1]

    if 'mpii' in datasets:
        # prepare mpii test for qualitative results experiments
        print('Processing MPII-TEST dataset...')
        mpii_extract(dataset_path=cfg.MPII_ROOT, out_path=cfg.DATASET_NPZ_PATH)

    if 'coco' in datasets:
        # prepare mpii test for qualitative results experiments
        print('Processing COCO-TEST dataset...')
        coco_extract(dataset_path=cfg.COCO_ROOT, out_path=cfg.DATASET_NPZ_PATH)

    if 'eft' in datasets:
        # convert eft data to npz format
        print('Processing EFT dataset...')
        eft_extract(dataset_path=cfg.EFT_ROOT, out_path=cfg.DATASET_NPZ_PATH)

    if '3dpw' in datasets:
        # create 3DPW all test set
        print('Processing 3DPW-ALL dataset...')
        pw3d_extract(dataset_path=cfg.PW3D_ROOT, out_path=cfg.DATASET_NPZ_PATH)

    if '3doh' in datasets:
        # create 3DOH50K training data
        print('Processing 3DOH-TRAIN dataset...')
        oh3d_train_extract(dataset_path=cfg.OH3D_ROOT, out_path=cfg.DATASET_NPZ_PATH)

        # create 3DOH50K training data
        print('Processing 3DOH-TEST dataset...')
        oh3d_test_extract(dataset_path=cfg.OH3D_ROOT, out_path=cfg.DATASET_NPZ_PATH)

    if 'mpi-inf-3dhp' in datasets:
        print('Processing MPI-INF-3DHP dataset...')
        mpi_inf_3dhp_extract(dataset_path=cfg.MPI_INF_3DHP_ROOT, openpose_path='',
                             out_path=cfg.DATASET_NPZ_PATH, mode='test')

    if 'ochuman' in datasets:
        print('Processing OCHUMAN dataset...')
        # missing_seq = [19, 28, 32, 49]
        ochuman_extract(dataset_path=cfg.OCHUMAN_ROOT, out_path=cfg.DATASET_NPZ_PATH,
                        split_idx=int(sys.argv[2]), num_splits=50)

    if 'crowdpose' in datasets:
        print('Processing CrowdPose dataset...')
        # missing_seq = [33, 34, 36, 49, 51, 55, 63, 65, 66, 69, 70, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99]
        crowdpose_extract(dataset_path=cfg.CROWDPOSE_ROOT, out_path=cfg.DATASET_NPZ_PATH,
                          split_idx=int(sys.argv[2]), num_splits=100)

    if 'aich' in datasets:
        print('Processing AICH dataset...')
        # missing_seq = [33, 34, 36, 49, 51, 55, 63, 65, 66, 69, 70, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99]
        aich_extract(dataset_path=cfg.AICH_ROOT, out_path=cfg.DATASET_NPZ_PATH,
                     split_idx=int(sys.argv[2]), num_splits=1)

    if 'muco3dhp' in datasets:
        print('Processing MuCo3DHP dataset...')
        muco3dhp_extract(dataset_path=cfg.MUCO3DHP_ROOT, out_path=cfg.DATASET_NPZ_PATH, augmented=True)