#CKPT1=logs/pare/03.09-pare_synth_occ_finetune/03-09-2020_15-46-30_03.09-pare_synth_occ_finetune/tb_logs_pare/0_4d1d75ce0c28412d980f9eaadc624f09/checkpoints/epoch=64.ckpt
#CKPT2=logs/pare/18.09-pare_cropaug_finetune/18-09-2020_14-02-23_18.09-pare_cropaug_finetune_train/tb_logs_pare/0_bb632709e67240f4b70a53adceeb797f/checkpoints/epoch=4.ckpt
#CFG=configs/pare_cfg.yaml
CKPT1=logs/spin/12.09-spin_synth_occ_finetune/12-09-2020_16-20-09_12.09-spin_synth_occ_finetune/tb_logs_pare/0_74211cc7bd3e44f79388c9ea8c5186c7/checkpoints/epoch=44.ckpt
CFG=configs/spin_cfg.yaml
BASE_DIR=logs/demo_results/videos/
VIDEOS="friends_bradpitt_0.25-0.37.mp4 friends_bradpitt.mp4 friends_mesh.mp4 friends_red_sweater.mp4"
#VIDEOS=" "

# vid=friends_bradpitt.mp4
# python demo.py --cfg $CFG --ckpt $CKPT2 --exp cropaug --vid_file $BASE_DIR$vid --no_save

for v in $VIDEOS
do
   echo "Running the demo for video $BASE_DIR$v"
   # python demo.py --cfg $CFG --ckpt $CKPT2 --exp cropaug --vid_file $BASE_DIR$v --no_save
   python demo.py --cfg $CFG --ckpt $CKPT1 --exp spin --vid_file $BASE_DIR$v --no_save
#   python demo.py --cfg $CFG --ckpt $CKPT1 --exp wo_cropaug_smooth --vid_file $BASE_DIR$v --no_save --smooth
done

# combine videos
# ffmpeg -i input_1.mp4 -i input_2.mp4 -filter_complex hstack output.mp4
# ffmpeg -i input.mp4 -ss 00:00:30.0 -to 00:00:37.0 output.mp4

# https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg
# ffmpeg -i input0 -i input1 -i input2 -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" output
# ffmpeg -i 1.mp4 -i 2.mp4 -i 3.mp4 -filter_complex hstack=3 out.mp4