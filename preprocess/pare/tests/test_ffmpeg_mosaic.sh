#ffmpeg -i logs/demo_results/atlas_pare-hrnet/atlas_pare-hrnet_result.mp4 -i logs/demo_results/atlas_spin/atlas_spin_result.mp4 -i logs/demo_results/atlas_eft/atlas_eft_result.mp4 -i logs/demo_results/atlas_vibe/atlas_vibe_result.mp4 -filter_complex "[0]drawtext=text='vid0':fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2[v0];[1]drawtext=text='vid1':fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2[v1];[2]drawtext=text='vid2':fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2[v2];[3]drawtext=text='vid3':fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2[v3];[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" logs/demo/demo_benchmark_videos/atlas_2x2_mosaic.mp4
DDIR=logs/demo/demo_results
ODIR=logs/demo/demo_benchmark_videos

#ffmpeg -threads 0 -y -i $DDIR/$1_spin/$1_spin_result.mp4 -i $DDIR/$1_vibe/$1_vibe_result.mp4 -i $DDIR/$1_eft/$1_eft_result.mp4 -i $DDIR/$1_pare-hrnet/$1_pare-hrnet_result.mp4 -filter_complex \
#      "[0]drawtext=text='SPIN':    fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [0:v];
#       [1]drawtext=text='VIBE':    fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [1:v];
#       [2]drawtext=text='HMR-EFT': fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [2:v];
#       [3]drawtext=text='PARE':    fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [3:v];
#       [0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
#      -map "[v]" $ODIR/$1_2x2_mosaic.mp4

ffmpeg -threads 0 -y -i $DDIR/$1_vibe/$1_vibe_result.mp4 -i $DDIR/$1_pare-hrnet/$1_pare-hrnet_result.mp4 -filter_complex \
      "[0]drawtext=text='VIBE':    fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [0:v];
       [1]drawtext=text='PARE':    fontfile=/usr/share/fonts/truetype/gentium-basic/GenBasB.ttf: fontcolor=white: fontsize=w/30: x=text_w/2: y=text_h [1:v];
       [0:v][1:v]hstack=inputs=2[v]" \
      -map "[v]" $ODIR/$1_vibe_vs_pare.mp4


# FONTFILES
# /usr/share/fonts/truetype/gentium-basic/GenBasB.ttf
# /usr/share/fonts/truetype/gentium/Gentium-R.ttf