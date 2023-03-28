LOGDIR=$1
VAL_DATASETS="3doh mpi-inf-3dhp 3dpw-all h36m-p2"
GPU_MEM=20000

for v in $VAL_DATASETS
do
   echo "Running Evaluation Script for $LOGDIR - $v"
   python eval.py --cfg $1/config_to_run.yaml \
                  --opts TESTING.SAVE_IMAGES False TRAINING.RESUME None DATASET.VAL_DS $v TRAINING.PRETRAINED_LIT python eval.py --cfg logs/pare_coco/29.09-spin_simpler_training_pretrained2d/29-09-2020_09-58-57_29.09-spin_simpler_training_pretrained2d_train/config_to_run.yaml --opts DATASET.VAL_DS 3doh TRAINING.PRETRAINED_LIT logs/pare_coco/29.09-spin_simpler_training_pretrained2d/29-09-2020_09-58-57_29.09-spin_simpler_training_pretrained2d_train/tb_logs_pare_coco/0_ce0dddb32fe348fa9adb5475c8ed349d/checkpoints/epoch=140.ckpt \
                  --cluster \
                  --gpu_min_mem $GPU_MEM
done