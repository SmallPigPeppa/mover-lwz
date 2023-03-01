#!/bin/bash
batch_size=1
img_list=-1
DATA_FOLDER="/share/wenzhuoliu/code/mover-lwz-fit3d/samples_clean_gta/FPS-5-clean-debug"
OUTPUT_FOLDER="/share/wenzhuoliu/code/mover-lwz-fit3d/out-data-gta-debug"
CALIBRATION_FOLDER=/share/wenzhuoliu/code/mover-lwz-fit3d/smplifyx_cam/
CONFIG_FILE=/share/wenzhuoliu/code/mover/body_models/cfg_files/fit_smplx_3D.yaml
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER=/share/wenzhuoliu/code/smplifyx/models
VPOSER_FOLDER=/share/wenzhuoliu/code/smplifyx/vposer_v1_0
conda activate mover

python main.py \
    --single "False" \
    --dataset "Pose2Room" \
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
    --part_segm_fn /share/wenzhuoliu/code/smplifyx/smplx_parts_segm.pkl \
    --camera_type "user" \
    --gender 'male' \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 3 \
    --end_opt_stage 4

