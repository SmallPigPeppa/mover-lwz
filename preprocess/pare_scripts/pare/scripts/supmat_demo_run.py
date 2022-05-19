import os

PARE_DIR = 'logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train/'
SPIN_DIR = 'logs/pare_coco/05.11-spin_ckpt_eval'
HMR_DIR = 'logs/pare/27.10-spin_all_data/27-10-2020_21-05-03_27.10-spin_all_data_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train'

PARE_CKPT = 'tb_logs_pare/0_1111239d5702427e8088b30d8584f263/checkpoints/epoch=0.ckpt'
SPIN_CKPT = ''
HMR_CKPT = ''

def main():
    vid_file = ''
    'python demo.py --exp pare_hrnet --vid_file logs/demo_results/videos/mission_impossible.mp4 --display --no_save'
    cmd = f'python demo.py --cfg {os.path.join(PARE_DIR, "config_to_run.yaml")} ' \
          f'--ckpt {os.path.join(PARE_DIR, PARE_CKPT)}' \
          f'--exp pare_hrnet --no_save' \
          f'--vid_file {vid_file}'

    os.system(cmd)

if __name__ == '__main__':
    main()