#!/bin/bash

dirA="/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/mover-out/FPS-5"
dirB="/mnt/mmtech01/dataset/vision_text_pretrain/gta-im/FPS-5"

for subdir in "$dirA"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        src_file="$subdir/results/001_all.pkl"
        dest_dir="$dirB/$subdir_name"
        dest_file="$dest_dir/info_smpl.pkl"
        if [ -e "$src_file" ] && [ -d "$dest_dir" ]; then
            cp "$src_file" "$dest_file"
        fi
    fi
done