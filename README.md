# MOVER Preprocess

1. openpose filter
2. pare
3. op2smplifyx_withPARE

## 
## Enviroment
prepare enviroment according to https://github.com/mkocabas/PARE
## Download pretrained model
1. data
2. smplifyx-file
3. hrnet_model
4. smplifyx_cam


## Prepare input data

input data directory organized like this
```
input-data/
├── Color_flip
│   ├── Color_flip.mp4
│   ├── images
│   └── openpose
└── test_video1
    ├── images
    ├── openpose
    └── test_video1.mp4
```
## Run preprocess for MOVER
modify input_dir out_dir and video_name in run.sh

```. run.sh```
