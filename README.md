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
5. vposer_v1_0

directory should be organized like this
```
mover-project-dir
├── hrnet_model
├── input-data
├── out-data
├── preprocess
├── README.md
├── run.sh
├── smplifyx_cam
├── smplifyx-file
└── vposer_v1_0
```


## Prepare input data

input data directory should be organized like this
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

output directory will be organized like this
```
out-data/
└── test_video1
    ├── op2smplifyx_withPARE
    ├── op2smplifyx_withPARE_ori_OP
    ├── op2smplifyx_withPARE_PARE3DJointOneConfidence_OP2DJoints
    ├── openpose_OneEurofilter
    └── pare
```