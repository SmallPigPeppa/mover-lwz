# run OneEuro Filter
video_n=Color_flip
input_dir=$1
#input_dir=/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02
# ${video_n}_rename_openpose
# root_use=/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/mv_smplifyx_input_withPARE_PARE3DJointOneConfidence_OP2DJoints
img_dir=${input_dir}/Color_flip_rename
root_dir=${input_dir}/Color_filp_rename_openpose
save_dir=${input_dir}/Color_flip_rename_openpose_OneEurofilter
source activate hdsr_new_bvh
python /is/cluster/hyi/workspace/HCI/footskate_reducer/ground_detector/op_filter_json.py \
        --root=${root_dir} \
        --dump=${save_dir} \
        --img_dir=${img_dir} \
        --viz=True \

