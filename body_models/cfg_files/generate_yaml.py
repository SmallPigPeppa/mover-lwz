from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import sys
import yaml
import os.path as osp
sys.path.append('/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo')
from body_models.smplifyx.cmd_parser import parse_config



import sys
import os

# import configargparse
# import ast 

# arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

# cfg_parser = configargparse.YAMLConfigFileParser
# description = 'PyTorch implementation of SMPLifyX'
# parser = configargparse.ArgParser(formatter_class=arg_formatter,
#                                     config_file_parser_class=cfg_parser,
#                                     description=description,
#                                     prog='SMPLifyX')
# args = parser.parse_args()
# args = vars(args)

# args = parse_config()
# args.pop('body_tri_idxs')

ori_cfg = sys.argv[1]

with open(ori_cfg, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
args = data_loaded

# import pdb;pdb.set_trace()
output_folder = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo/body_models/cfg_files/cluster'
os.makedirs(output_folder, exist_ok=True)

# scene_loss_weight = [1e-2, 1e0, 1e-3, 1e0, 1e-4, 1e0, 1e-2, 1e1, 1e-2, 1e-1, 1e-2, 5e-1, 1e-2, 1e-2]
# scene_loss_weight = []
# on 01.27:
# ordinal_depth_loss_weight = [2e-2, 2e-4, 2e-1, 2e-4, 2e0, 2e-4, 2e-3, 2e-4, 2e-4, 2e-4] #10
# sdf_penetration_loss_weight = [5e-1, 5e-1, 5e-2, 5e-2, 5e-3, 5e-3, 5e-4, 5e-4, 5e0, 5e0] # 10
# contact_loss_weights = [1e4, 1e4, 5e4, 5e4, 1e5, 1e5, 1e6, 1e6, 1e3, 1e3, 1e2, 1e2, 1e1, 1e1, 1e-1, 1e-1] #16 

# on 01.27 10:00
# ordinal_depth_loss_weight = [2e-1, 2e-2, 2e-1, 2e-1, 2e-3, 2e-3, 2e-2, 2e-4, 2e-1, 2e-4, 2e0, 2e-4, 2e-3, 2e-4, 2e-4, 2e-4] #10 | 2e-4, 2e-4
# sdf_penetration_loss_weight = [5e-1, 5e-1, 5e-2, 5e-2, 5e-3, 5e-3, 5e-4, 5e-4, 5e0, 5e0] # 10
# ordinal_depth_loss_weight = [2e0, 2e0, 2e-1, 2e-1, 2e-2, 2e-2, 2e-3, 2e-3, 2e-4, 2e-4] #10 | 2e-4, 2e-4
# sdf_penetration_loss_weight = [5e1, 5e1, 5e0, 5e0, 5e-1, 5e-1, 5e-2, 5e-2] #5e-2 very slightly, but useful; use

# on 01.30
# ordinal_depth_loss_weight = [2e-1, 2e-1] # 2e1, 2e1, 2e2, 2e2 #10 | 2e-4, 2e-4
# sdf_penetration_loss_weight = [5e4, 5e4] #5e2, 5e2, 5e3, 5e3,
# contact_loss_weights = [] #16 1e4, 1e4, 1e3, 1e3

# on 01.31

contact_loss_weights = [1e6, 1e5, 1e4] #1e6, 1e5, 1e4, 1e3, 1e3
sdf_penetration_loss_weight = [5000, 2000, 1000, 500] #5e2, 5e2, 5e3, 5e3,

start = 0

for i in range(int(len(contact_loss_weights))):
    for j in range(int(len(sdf_penetration_loss_weight))):
        args['contact_loss_weights'][-3] = 0
        args['contact_loss_weights'][-2] = 0
        args['contact_loss_weights'][-1] = contact_loss_weights[i]
        # args['ordinal_depth_loss_weight'][-2] = 0
        # args['ordinal_depth_loss_weight'][-1] = 0
        args['sdf_penetration_loss_weight'][-2] = sdf_penetration_loss_weight[j]
        # args['sdf_penetration_loss_weight'][-1] = 0`
        conf_fn = osp.join(output_folder, f'conf_{start}.yaml')
        with open(conf_fn, 'w') as conf_file:
            yaml.dump(args, conf_file)
        start += 1
    
# for j in range(int(len(ordinal_depth_loss_weight)/2)):
#     args['contact_loss_weights'][-2] = 0
#     args['contact_loss_weights'][-1] = 0
#     args['ordinal_depth_loss_weight'][-2] = ordinal_depth_loss_weight[j*2+0]
#     args['ordinal_depth_loss_weight'][-1] = ordinal_depth_loss_weight[j*2+1]
#     args['sdf_penetration_loss_weight'][-2] = 0
#     args['sdf_penetration_loss_weight'][-1] = 0
#     conf_fn = osp.join(output_folder, f'conf_{start}.yaml')
#     with open(conf_fn, 'w') as conf_file:
#         yaml.dump(args, conf_file)
#     start += 1

# for k in range(int(len(sdf_penetration_loss_weight)/2)):
#     args['contact_loss_weights'][-2] = 0
#     args['contact_loss_weights'][-1] = 0
#     args['ordinal_depth_loss_weight'][-2] = 0
#     args['ordinal_depth_loss_weight'][-1] = 0
#     args['sdf_penetration_loss_weight'][-2] = sdf_penetration_loss_weight[k*2+0]
#     args['sdf_penetration_loss_weight'][-1] = sdf_penetration_loss_weight[k*2+1]
#     conf_fn = osp.join(output_folder, f'conf_{start}.yaml')
#     with open(conf_fn, 'w') as conf_file:
#         yaml.dump(args, conf_file)
#     start += 1