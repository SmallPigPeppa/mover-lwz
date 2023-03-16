#!/bin/bash
batch_size=1
img_list=-1
DATA_FOLDER="/share/wenzhuoliu/code/mover-lwz-fit3d-smpl/samples_clean_gta/FPS-5-clean-debug"
OUTPUT_FOLDER="/share/wenzhuoliu/code/mover-lwz-fit3d-smpl-all/out-data-gta"
CALIBRATION_FOLDER= "./smplifyx_cam/"
CONFIG_FILE= "./body_models/cfg_files/fit_smpl_3D.yaml"
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER= "./models"
VPOSER_FOLDER= "./vposer_v1_0"
PART_SEGM_FN= "./smplx_parts_segm.pkl"
#conda activate mover
#cd /share/wenzhuoliu/code/mover-lwz-fit3d-smpl
python main.py \
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
    --model_type 'smpl' \
    --pre_load="False" \
    --pre_load_pare_pose="False" \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn ${PART_SEGM_FN} \
    --camera_type "user" \
    --gender 'neutral' \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 3 \
    --end_opt_stage 4

