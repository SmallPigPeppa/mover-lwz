import json

data = json.load(open('/is/cluster/work/mkocabas/datasets/eft/eft_fit/LSPet_ver01.json'))

annots = data['data']

# dict_keys [
# 'parm_pose',
# 'parm_shape',
# 'parm_cam',
# 'bbox_scale',
# 'bbox_center',
# 'gt_keypoint_2d',
# 'joint_validity_openpose18',
# 'smpltype',
# 'annotId',
# 'imageName'
# ]

# parm_pose (24, 3, 3)
# parm_shape (10,)
# parm_cam (3,)
# bbox_center (2,)
# gt_keypoint_2d (49, 3)
# joint_validity_openpose18 (18,)

import IPython; IPython.embed(); exit(1)