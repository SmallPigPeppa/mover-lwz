#!/bin/bash
export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
echo $EGL_DEVICE_ID
echo $1
pid=$(printf "%06d" $1)
echo ${pid}
DATA_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_input_pare3d_opfeetAnkles/${pid}/00/"
OUTPUT_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_result_pare3d_opfeetAnkles/${pid}"
#DATA_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_input_pare3d/${pid}/00/"
#OUTPUT_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_result_pare3d/${pid}"
# DATA_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_input/${pid}/00/"
# OUTPUT_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_result_persp/${pid}"
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER=/lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/models
VPOSER_FOLDER=/lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/vposer_v1_0
CALIBRATION_FOLDER=/ps/scratch/hyi/HCI_dataset/holistic_scene_human/smplifyx_test/xml
source /is/cluster/hyi/venv/smplify/bin/activate
python /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smplifyx/main_mv.py \
    --config /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/cfg_files/fit_smplx.yaml \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --visualize="False" \
    --model_folder ${MODEL_FOLDER} \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/smplx_parts_segm.pkl \
    --camera_type "user" \
    --gender male \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 2 \