import os
import shutil
import glob
import json
import numpy as np
from tqdm import tqdm
img_dir = '/ps/scratch/hyi/HCI_dataset/holistic_scene_human/image_high_resolution'
json_dir = '/ps/scratch/hyi/HCI_dataset/holistic_scene_human//openpose_result'
output_dir = '/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_input_withfixed3d'

# img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

def modify_json(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)
    # import pdb;pdb.set_trace()
    # results['people'][0]['pose_keypoints_3d'] = np.zeros((25, 4)).reshape(-1).tolist()
    results['people'][0]['pose_keypoints_3d'] = np.array([-0.22412225604057312, -0.6498373746871948, 2.9610018730163574, 
            0.0, 1.248719573020935, -1.0115410089492798, 3.6479299068450928, 0.0, -0.0635649710893631, -0.5221863389015198, 
            3.206291437149048, 0.0, -0.027896245941519737, -0.48878908157348633, 3.4668233394622803, 0.0, -0.0032307107467204332,
             -0.5475852489471436, 3.714127540588379, 0.0, -0.029576370492577553, -0.4711841642856598, 2.8522140979766846, 0.0, 0.051362618803977966, 
             -0.34874120354652405, 2.645429849624634, 0.0, 0.018674185499548912, -0.3764256238937378, 2.406602144241333, 0.0, 0.679316520690918, -1.2989616394042969, 
             4.081627368927002, 0.0, -0.06682337820529938, 0.043380845338106155, 3.1829376220703125, 0.0, -0.05268491804599762, 0.47060418128967285, 3.2917580604553223,
              0.0, -0.02126387506723404, 0.8698357343673706, 3.3812973499298096, 0.0, -0.08949077129364014, 0.07104416936635971, 2.985130548477173, 0.0,
               -0.047359202057123184, 0.5159651041030884, 3.022535800933838, 0.0, -0.003500135848298669, 0.9128473401069641, 3.057770252227783, 0.0,
                -0.21456217765808105, -0.6910736560821533, 2.9972853660583496, 0.0, -0.1919327974319458, -0.6841851472854614, 2.9299850463867188, 0.0, 
                -0.14674854278564453, -0.6980823278427124, 3.0769851207733154, 0.0, -0.13207750022411346, -0.6670486330986023, 2.879300594329834, 0.0, 
                -0.18096306920051575, 0.9664211273193359, 2.99873685836792, 0.0, -0.12371698021888733, 0.9613545536994934, 2.9702961444854736, 0.0, 
                0.027623290196061134, 0.9609692692756653, 3.082742691040039, 0.0, -0.2177031934261322, 0.9018409848213196, 3.4231624603271484, 0.0, 
                -0.1774619221687317, 0.8976943492889404, 3.456867218017578, 0.0, 0.021769778802990913, 0.925752580165863, 3.3631489276885986, 1.0]).tolist()
    return results

for one in tqdm(sorted(os.listdir(img_dir))):
    # import pdb;pdb.set_trace()
    base, ext = os.path.splitext(one)
    os.makedirs(os.path.join(output_dir, base, '00', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, base, '00', 'keypoints'), exist_ok=True)

    ori_img_p = os.path.join(img_dir, f'{one}')
    dst_path = os.path.join(output_dir, base, '00', 'images', f'{one}')
    if not os.path.exists(dst_path):
        os.system(f'ln -s {ori_img_p} {dst_path}')

    ori_json = os.path.join(json_dir, f'{base}_keypoints.json')
    new_result = modify_json(ori_json)
    dst_json = os.path.join(output_dir, base, '00', 'keypoints', f'{base}_keypoints.json')
    with open(dst_json, 'w') as f:
        json.dump(new_result, f)