#!/bin/bash

batch_size=1
img_list=-1
PROJECT_PATH="/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all"

DATA_FOLDER="${PROJECT_PATH}/mover-input/FPS-30/"
OUTPUT_FOLDER="${PROJECT_PATH}/mover-out/FPS-30"
CALIBRATION_FOLDER="${PROJECT_PATH}/smplifyx_cam/"
CONFIG_FILE="${PROJECT_PATH}/body_models/cfg_files/fit_smpl_3D.yaml"
MODEL_FOLDER="${PROJECT_PATH}/models"
VPOSER_FOLDER="${PROJECT_PATH}/vposer_v1_0"
PART_SEGM_FN="${PROJECT_PATH}/smplx_parts_segm.pkl"

echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}

#conda activate mover
#cd /share/wenzhuoliu/code/mover-lwz-fit3d-smpl

for REC_IDX in $(ls -d ${DATA_FOLDER}*/ | xargs -n 1 basename); do
  REC_DATA_FOLDER="${DATA_FOLDER}${REC_IDX}"

  echo "Processing ${REC_IDX}"

  /root/miniconda3/envs/mover/bin/python main.py \
      --single False \
      --dataset Pose2Room \
      --config ${CONFIG_FILE} \
      --img_list ${img_list} \
      --batch_size ${batch_size} \
      --data_folder ${REC_DATA_FOLDER} \
      --output_folder ${OUTPUT_FOLDER} \
      --visualize False \
      --save_meshes False \
      --model_folder ${MODEL_FOLDER} \
      --model_type smpl \
      --pre_load False \
      --pre_load_pare_pose False \
      --vposer_ckpt ${VPOSER_FOLDER} \
      --part_segm_fn ${PART_SEGM_FN} \
      --camera_type user \
      --gender neutral \
      --calib_path ${CALIBRATION_FOLDER} \
      --start_opt_stage 3 \
      --end_opt_stage 4
done
