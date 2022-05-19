#!/bin/bash
# export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
# echo $EGL_DEVICE_ID
# pid=$(printf "%06d" $1)
# echo ${pid}
# modify
batch_size=100
# img_list=`expr "$1" + "2"`
img_list=$1
INPUT_DATA="/is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/debug/smplify_video_input/"
DATA_FOLDER=${INPUT_DATA}/"mv_smplifyx_input_OneEuroFilter_PARE_PARE3DJointOneConfidence_OP2DJoints"
OUTPUT_FOLDER=${INPUT_DATA}/"results"
CALIBRATION_FOLDER=${INPUT_DATA}/smplifyx_cam
CONFIG_FILE=/is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/body_models/cfg_files/fit_smplx_video.yaml
# end of modify
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
# --cam_inc_fn ${cam_inc_fn} \
MODEL_FOLDER=/is/cluster/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/models
VPOSER_FOLDER=/is/cluster/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/vposer_v1_0
source /is/cluster/hyi/venv/smplify/bin/activate
# first opt 3D joint on 3D skeletons
# Then opt in [start_opt_stage, end_opt_stage): focus on hands.

# cd /is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/
# export PYTHONPATH=/is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify:${PYTHONPATH}
# python body_models/video_smplifyx/main_video.py \
# ! save_meshes=True: save mesh and rendered images.
python /is/cluster/hyi/workspace/SceneGeneration/smplify-x_modify/main.py \
    --single "False" \
    --config ${CONFIG_FILE} \
    --img_list ${img_list} \
    --batch_size ${batch_size} \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --visualize="False" \
    --save_meshes=True \
    --model_folder ${MODEL_FOLDER} \
    --model_type 'smplx' \
    --pre_load="False" \
    --pre_load_pare_pose="False" \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/smplx_parts_segm.pkl \
    --camera_type "user" \
    --gender 'male' \
    --use_video "True" \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 3 \
    --end_opt_stage 5 \