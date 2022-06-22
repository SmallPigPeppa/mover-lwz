python demo.py \
       --cfg logs/pare_coco/28.11-pare_mobilenet/28-11-2020_23-05-33_28.11-pare_mobilenet_train/config_to_run.yaml \
       --ckpt logs/pare_coco/28.11-pare_mobilenet/28-11-2020_23-05-33_28.11-pare_mobilenet_train/tb_logs_pare_coco/0_b12629aa3c7e485681849d6a9610c03b/checkpoints/epoch=86.ckpt \
       --exp pare_mobilenet --vid_file $1 \
       --detector maskrcnn