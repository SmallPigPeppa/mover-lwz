python demo.py \
       --cfg logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train/config_to_run.yaml \
       --ckpt logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train/tb_logs_pare/0_1111239d5702427e8088b30d8584f263/checkpoints/epoch=0.ckpt \
       --exp pare_hrnet --vid_file $1 \
       --detector maskrcnn