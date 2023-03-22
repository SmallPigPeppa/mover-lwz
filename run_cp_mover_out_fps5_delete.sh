#!/bin/bash

dirA="/mnt/mmtech01/usr/liuwenzhuo/code/mover-lwz-fit3d-smpl-all/mover-out/FPS-5"
dirB="/mnt/mmtech01/dataset/vision_text_pretrain/gta-im/FPS-5"

for subdir in "$dirB"/*; do
    if [ -d "$subdir" ]; then
        target_file="$subdir/001_all.pkl"
        if [ -e "$target_file" ]; then
            rm "$target_file"
        fi
    fi
done