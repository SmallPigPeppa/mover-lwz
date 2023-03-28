cd /is/cluster/work/mkocabas/projects/mmpose

python demo/top_down_img_demo_with_mmdet.py ../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/hrnet/coco-wholebody/hrnet_w48_coco_wholebody_256x192.py \
    checkpoints/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth \
    --img-root $1 \
    --out-img-root $2/mmpose_results/ --save
