video_n=Color_flip
input_dir=/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02
save_dir=/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/quantitative_ori_flip/ground_detect
python op2npy.py ${input_dir}/${video_n}_rename_openpose ${save_dir}
vid_path=${input_dir}/${video_n}.mp4
# openpose result need preprocess while it fails, currently, directly copy previous frame result to the failed image
op_path=${save_dir}/${video_n}_rename_openpose/${video_n}_rename_openpose.npy
out_dir=${save_dir}
python inference.py --vid_path ${vid_path} \
                    --op_path ${op_path} \
                    --out_dir ${out_dir} \
