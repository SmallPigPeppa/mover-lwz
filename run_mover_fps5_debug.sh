
batch_size=1
img_list=-1
#PROJECT_PATH="/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all"
PROJECT_PATH="/share/wenzhuoliu/code/mover-lwz-fit3d-smpl-all"
REC_IDX="2020-06-09-16-09-56"


DATA_FOLDER="${PROJECT_PATH}/mover-input/FPS-5/${REC_IDX}"
OUTPUT_FOLDER="${PROJECT_PATH}/mover-out"
CALIBRATION_FOLDER="${PROJECT_PATH}/smplifyx_cam/"
CONFIG_FILE="${PROJECT_PATH}/body_models/cfg_files/fit_smpl_3D.yaml"
MODEL_FOLDER="${PROJECT_PATH}/models"
VPOSER_FOLDER="${PROJECT_PATH}/vposer_v1_0"
PART_SEGM_FN="${PROJECT_PATH}/smplx_parts_segm.pkl"

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


