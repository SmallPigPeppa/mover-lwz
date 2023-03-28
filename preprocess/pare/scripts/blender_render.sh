/home/mkocabas/Desktop/blender-2.80-linux-glibc217-x86_64/blender -b data/pare_render.blend \
  --python pare/utils/blender.py -- \
  -i $1 \
  -o $1 \
  -w -t 0.1 --sideview -c $2 -s $3 $4