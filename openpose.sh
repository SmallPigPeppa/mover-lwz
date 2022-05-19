
openpose_dir="C:\Users\Administrator\Desktop\openpose"
video_name="color_flip"
input_dir="${openpose_dir}/input_video/${video_name}"
output_dir="${openpose_dir}/openpose_out/${video_name}"
# video
# ${openpose_dir}\bin\OpenPoseDemo.exe --video ${input_dir} --face --hand --write_json ${output_dir}
# image
${openpose_dir}\bin\OpenPoseDemo.exe --image_dir ${input_dir} --face --hand --write_json ${output_dir}


