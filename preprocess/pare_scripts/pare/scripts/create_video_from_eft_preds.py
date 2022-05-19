import os
import sys

from pare.utils.demo_utils import images_to_video, concat_videos

exp_dir = sys.argv[1]

inp_dir = f'{exp_dir}/output_images'
eft_vid = f'{exp_dir}/video_eft.mp4'
images_to_video(inp_dir, eft_vid, img_suffix='%05d_ft.jpg')


init_vid = f'{exp_dir}/video_init.mp4'
images_to_video(inp_dir, init_vid, img_suffix='%05d_init.jpg')

concat_vid = f'{exp_dir}/video_concat.mp4'
concat_videos(video_list=[init_vid, eft_vid], output_f=concat_vid)