# PARE: Pixel Aligned SMPL Regression

## Features

## Getting Started

- Clone the repo. We will call the cloned direcory `$ROOT`
- Install the requirements using [`requirements.txt`](requirements.txt)
- Link or copy this folder `/ps/scratch/ps_shared/mkocabas/pare_results/data` to your project as `$ROOT/data`.


## Demo

### Video Demo
Run the command below. See `demo.py` for more input options.
```shell script
export DEMO_DATA=/ps/scratch/ps_shared/mkocabas/pare_results/demo_data/hrnet_model
python demo.py \
       --cfg $DEMO_DATA/config.yaml \
       --ckpt $DEMO_DATA/checkpoint.ckpt \
       --vid_file $DEMO_DATA/sample_video.mp4 \
       --draw_keypoints
```

**New**: `draw_keypoints` argument draw 2d joints on rendered image.

### Image Folder Demo

Coming soon...

### Webcam Demo

Coming soon...

#### Output format

If demo finishes succesfully, it needs to create a file named `vibe_output.pkl` in the `--output_folder`.
We can inspect what this file contains by:

```
>>> import joblib # you may use native pickle here as well

>>> output = joblib.load('pare_output.pkl') 

>>> print(output.keys())  
                                                                                                                                                                                                                                                                                                                                                                                              
dict_keys([1, 2, 3, 4]) # these are the track ids for each subject appearing in the video

>>> for k,v in output[1].items(): print(k,v.shape) 

pred_cam (n_frames, 3)          # weak perspective camera parameters in cropped image space (s,tx,ty)
orig_cam (n_frames, 4)          # weak perspective camera parameters in original image space (sx,sy,tx,ty)
verts (n_frames, 6890, 3)       # SMPL mesh vertices
pose (n_frames, 72)             # SMPL pose parameters
betas (n_frames, 10)            # SMPL body shape parameters
joints3d (n_frames, 49, 3)      # SMPL 3D joints
joints2d (n_frames, 21, 3)      # 2D keypoint detections by STAF if pose tracking enabled otherwise None
bboxes (n_frames, 4)            # bbox detections (cx,cy,w,h)
frame_ids (n_frames,)           # frame ids in which subject with tracking id #1 appears
smpl_joints2d (n_frames, 49, 2) # SMPL 2D joints 
```

## Running EFT on images
This demo runs with MMPose + MMDetection, please follow installation instructions: 
- [mmpose](https://github.com/open-mmlab/mmpose)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

After installation, modify [`pare/core/config.py#L43-44`](pare/core/config.py#L43-44) to point your mmpose/mmdetection directory.

If you have 2d joints for your images modify the script to be able to use them.
```shell script
python scripts/run_eft_on_images.py \
      --image_folder <path to image folder> \
      --log_dir logs/eft \
      --ckpt /ps/scratch/ps_shared/mkocabas/pare_results/spin_pretrained_ckpt_for_eft/epoch=77.ckpt
```

## Running SMPLify (with focal length optim)
```shell script
python scripts/run_smplify.py \
       --dataset 3dpw \ 
       --cam_step_size 0.1 \
       --cam_opt_iter 500 \
       --cam_opt_params camera_translation focal_length \
       --focal_length 750 \
       --batch_size 32 \
       --log \
       --save
```

Notes:

- Original smplify `cam_step_size` is 0.01, `cam_opt_iter` is 100, `focal_length` is 5000. 
- Other options for `--camp_opt_params` are `global_orient, camera_translation, focal_length, camera_rotation` 
- `save` options render the results images and save them in a log dir.

## Training

- Local

```shell script
python train.py --cfg configs/spin_cfg.yaml
```

- Cluster
```shell script
python train.py --cfg configs/spin_cfg.yaml --cluster --bid 300 --memory 16000 --num_cpus 8
```
Run above command in the login node. This script is standalone and automatic, 
you do not need to create a submission script.

Config management is designed to support grid search experiments.
You only need to give a list of values for a hyperparameter as input. Config will spawn
multiple experiments for each parameter defined as lists. This feature is effective on cluster only.

Example:
```yaml
...
DATASET:
  IMG_RES: [224, 256, 320]
  BATCH_SIZE: [32, 48, 64]
  NUM_WORKERS: 8
  PIN_MEMORY: True
  SHUFFLE_TRAIN: True
...
```

Config script will automatically run 9 different experiments for different combinations of hyperparams.


## Evaluation

Set `TRAINING.RESUME` to the pretrained checkpoint file you would like to evaluate, then run:

```shell script
python eval.py --cfg configs/spin_eval_cfg.yaml \
               --opts DATASET.VAL_DS 3dpw TRAINING.PRETRAINED_LIT <path to ckpt>
```

