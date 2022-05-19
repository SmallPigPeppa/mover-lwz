#!/bin/bash
export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
echo $EGL_DEVICE_ID
echo $1
# ! modify
# DATA_FOLDER='/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/Color_flip_rename'
# OUTPUT_FOLDER="/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/OneEuro_filter_mv_smplifyx_input_withPARE" 
# JSON_FOLDER="/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/openpose_filter/Color_flip_rename_openpose"
# pare_result='/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/pare/pare_output.pkl'
DATA_FOLDER=$1
OUTPUT_FOLDER=$2
JSON_FOLDER=$3
pare_result=$4
# not use: save 3D joints in camera CS.
cam_dir='/ps/scratch/hyi/HCI_dataset/holistic_scene_human/smplifyx_test/xml'
# ! end
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER=/ps/scratch/ps_shared/mkocabas/pare_results/data/body_models
VPOSER_FOLDER=/lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/vposer_v1_0
# export PATH=${PATH}
echo ${PATH}
#source /is/cluster/hyi/venv/smplify/bin/activate
cd /is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_pare_scripts/
# source activate pare_pt1.6_vision0.7
# which python
export PATH=/home/hyi/.local/bin:/home/hyi/anaconda3/envs/pare_pt1.6_vision0.7/bin:/home/hyi/anaconda3/condabin:/home/hyi/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/bin:/usr/bin
echo ${PATH}
module load cuda/10.0
/home/hyi/anaconda3/envs/pare_pt1.6_vision0.7/bin/python demo_pare_result.py \
    --config ./cfg_files/fit_smpl.yaml \
    --export_mesh True \
    --save_new_json True \
    --json_folder  ${JSON_FOLDER} \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --pare_result ${pare_result} \
    --cam_dir ${cam_dir} \
    --visualize="False" \
    --model_folder ${MODEL_FOLDER} \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn /lustre/home/hyi/workspace/Multi-IOI/multiview_smplifyx/smpl-x_model/smplx_parts_segm.pkl \
    --gender male \
    --check_inverse_feet="False" \
