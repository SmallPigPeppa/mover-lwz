##pip install git+https://github.com/rbgirshick/yacs.git
#export DEMO_DATA=/ps/scratch/ps_shared/mkocabas/pare_results/demo_data/hrnet_model
## source activate pare_pt1.6_vision0.7
#export PATH=${PATH}
#cd /is/cluster/hyi/workspace/HCI/hdsr/projects/pare/
#/home/hyi/anaconda3/envs/pare_pt1.6_vision0.7/bin/python demo.py \
#       --cfg $DEMO_DATA/config.yaml \
#       --ckpt $DEMO_DATA/checkpoint.ckpt \
#       --output_folder $2 \
#       --vid_file $1 \
#       --draw_keypoints \
#       --detector maskrcnn
#
##--vid_file /ps/scratch/hyi/HCI_dataset/holistic_scene_human/pigraph_input_image_high_resolution.mp4 \
## tmp_save_dir = f'{save_dir}/{video_name}/pare_results'
## {save_dir}/{video_name}/Color_flip.mp4




video_name=Color_flip
save_dir=/root/code/mover/preprocess
pare_model=/root/code/mover/preprocess/pare/hrnet_model

cd /root/code/mover/preprocess/pare/
/root/anaconda3/envs/mover2/bin/python demo.py \
       --cfg ${pare_model}/config.yaml \
       --ckpt ${pare_model}/checkpoint.ckpt \
       --output_folder $2 \
       --vid_file $1 \
       --draw_keypoints \
       --detector maskrcnn