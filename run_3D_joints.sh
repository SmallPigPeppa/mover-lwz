#!/bin/bash
# export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
# echo $EGL_DEVICE_ID
# pid=$(printf "%06d" $1)
# echo ${pid}
# modify
batch_size=1
# img_list=2
# img_list=`expr "$1" + "2"`
img_list=$1
DATA_FOLDER="/ps/project/scene_generation/Pose2Room/samples_clean"
OUTPUT_FOLDER="/is/cluster/work/hyi/results/SceneGeneration/Pose2Room"
CALIBRATION_FOLDER=/ps/scratch/hyi/HCI_dataset/holistic_scene_human/smplifyx_test/xml
CONFIG_FILE=/is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/body_models/cfg_files/fit_smplx_3D.yaml
# end of modify
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER=/is/cluster/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/models
VPOSER_FOLDER=/is/cluster/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/vposer_v1_0
source /is/cluster/hyi/venv/smplify/bin/activate
# first opt 3D joint on 3D skeletons
# Then opt in [start_opt_stage, end_opt_stage): focus on hands.

# cd /is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/
# export PYTHONPATH=/is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify:${PYTHONPATH}
# python body_models/video_smplifyx/main_video.py \
python /is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/main.py \
    --single "False" \
    --dataset "Pose2Room" \
    --config ${CONFIG_FILE} \
    --img_list ${img_list} \
    --batch_size ${batch_size} \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --visualize="False" \
    --save_meshes=False \
    --model_folder ${MODEL_FOLDER} \
    --model_type 'smplx' \
    --pre_load="False" \
    --pre_load_pare_pose="False" \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/smplx_parts_segm.pkl \
    --camera_type "user" \
    --gender 'male' \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 3 \
    --end_opt_stage 4 \