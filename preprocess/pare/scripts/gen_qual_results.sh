# SPIN
LOGDIR="logs/pare_coco/05.11-spin_ckpt_eval"
BATCH_SIZE=16
RENDER_RES=480
GPU_MEM=10000

#echo "Running Evaluation Scripts for $LOGDIR"
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS 3dpw-all_mpi-inf-3dhp DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM \
#              --no_best_ckpt
#
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS coco_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM \
#              --no_best_ckpt
#
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS 3doh_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM \
#              --no_best_ckpt

# HMR - EFT
#LOGDIR="logs/pare/27.10-spin_all_data/27-10-2020_21-05-03_27.10-spin_all_data_dataset.datasetsandratios-h36m_mpii_lspet_coco_mpi-inf-3dhp_0.5_0.3_0.3_0.3_0.2_train"
#
#echo "Running Evaluation Scripts for $LOGDIR"
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS 3dpw-all_mpi-inf-3dhp DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM
#
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS coco_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM
#
#python eval.py --cfg $LOGDIR/config_to_run.yaml \
#              --opts DATASET.VAL_DS 3doh_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True \
#              --cluster \
#              --gpu_min_mem $GPU_MEM


# PARE
LOGDIR="logs/pare/13.11-pare_hrnet_shape-reg_all-data/13-11-2020_19-07-37_13.11-pare_hrnet_shape-reg_all-data_dataset.datasetsandratios-h36m_coco_mpi-inf-3dhp_0.2_0.3_0.5_train"
GPU_MEM=30000
CKPT=$LOGDIR/tb_logs_pare/0_1111239d5702427e8088b30d8584f263/checkpoints/epoch=0.ckpt

echo "Running Evaluation Scripts for $LOGDIR"
python eval.py --cfg $LOGDIR/config_to_run.yaml \
              --opts DATASET.VAL_DS 3dpw-all_mpi-inf-3dhp DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True TRAINING.PRETRAINED_LIT $CKPT \
              --cluster \
              --gpu_min_mem $GPU_MEM

python eval.py --cfg $LOGDIR/config_to_run.yaml \
              --opts DATASET.VAL_DS coco_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True TRAINING.PRETRAINED_LIT $CKPT \
              --cluster \
              --gpu_min_mem $GPU_MEM

python eval.py --cfg $LOGDIR/config_to_run.yaml \
              --opts DATASET.VAL_DS 3doh_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True TRAINING.PRETRAINED_LIT $CKPT \
              --cluster \
              --gpu_min_mem $GPU_MEM


python eval.py --cfg $LOGDIR/config_to_run.yaml \
              --opts DATASET.VAL_DS 3dpw-all_mpii DATASET.RENDER_RES $RENDER_RES DATASET.BATCH_SIZE $BATCH_SIZE TESTING.SAVE_IMAGES True TESTING.SAVE_MESHES True TESTING.SIDEVIEW True TRAINING.PRETRAINED_LIT $CKPT \
              --cluster \
              --gpu_min_mem $GPU_MEM