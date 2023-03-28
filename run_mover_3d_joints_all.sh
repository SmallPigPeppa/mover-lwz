#!/bin/bash
batch_size=1
img_list=-1
DATA_FOLDER=/share/wenzhuoliu/code/mover-lwz-fit3d-smpl/samples_clean_gta/FPS-5-clean-debug
OUTPUT_FOLDER=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/mover-out
CALIBRATION_FOLDER=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/smplifyx_cam/
CONFIG_FILE=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/body_models/cfg_files/fit_smpl_3D.yaml
MODEL_FOLDER=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/models
VPOSER_FOLDER=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/vposer_v1_0
PART_SEGM_FN=/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/smplx_parts_segm.pkl

echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
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
    --save_meshes=True \
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

