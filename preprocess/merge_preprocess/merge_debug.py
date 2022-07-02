# from demo_merge import main0
import argparse
from merge_parser import parser_pare, args_pare_result

if __name__ == "__main__":
    parser1 = parser_pare()
    pare_model = '/root/code/mover/preprocess/pare/hrnet_model'
    out_dir = '/root/code/mover/preprocess/merge_preprocess/Color_flip-out'
    video_file='/root/code/mover/preprocess/pare/Color_flip.mp4'
    args1 = parser1.parse_args()

    # args2= args_pare_result()

    print(args1)
    parser1.set_defaults(cfg=f'{pare_model}/config.yaml', ckpt=f'{pare_model}/checkpoint.ckpt',
                         output_folder=f'{out_dir}',vid_file=f'${video_file}',draw_keypoints=True,detector='maskrcnn')
    args1 = parser1.parse_args()

    # args2= args_pare_result()

    print(args1)
    # print(args1.output_folder)

# args=vars(args)
# print(args)

# def f(**args):
#     print(args.output_folder)
#
# f(**args)
# main0(args)
