from demo_merge import main0
from demo_pare_result_merge import main1
import argparse
from merge_parser import parser_pare, parser_pare_result

if __name__ == "__main__":
    parser1 = parser_pare()
    pare_model = '/root/code/mover/preprocess/pare/hrnet_model'
    out_dir = '/root/code/mover/preprocess/merge_preprocess/Color_flip-out'
    video_file = '/root/code/mover/preprocess/pare/Color_flip.mp4'
    parser1.set_defaults(cfg=f'{pare_model}/config.yaml', ckpt=f'{pare_model}/checkpoint.ckpt',
                         output_folder=f'{out_dir}', vid_file=f'${video_file}', draw_keypoints=True,
                         detector='maskrcnn')
    args1 = parser1.parse_args()
    print(args1)
    print('********************************************')

    parser2 = parser_pare_result()
    args2 = parser2.parse_args()
    args_dict2 = vars(args2)
    print(args_dict2)

    DATA_FOLDER = '/root/code/mover/preprocess/Color_flip/Color_flip_frames'
    OUTPUT_FOLDER = "/root/code/mover/preprocess/Color_flip_merge/OneEuro_filter_mv_smplifyx_input_withPARE"
    JSON_FOLDER = "/root/code/mover/preprocess/Color_flip/Color_flip_openpose"
    pare_result = '/root/code/mover/preprocess/Color_flip/pare_output.pkl'
    cam_dir = '/root/code/mover/smplifyx_cam'
    MODEL_FOLDER = '/root/code/mover/preprocess/pare/data/body_models'
    VPOSER_FOLDER = '/root/code/mover/smplifyx-file/vposer_v1_0'
    parser2.set_defaults(config='./cfg_files/fit_smpl.yaml', export_mesh=True, save_new_json=True,
                         json_folder=f'{JSON_FOLDER}', data_folder=f'{DATA_FOLDER}', output_folder=f'{OUTPUT_FOLDER}',
                         pare_result=f'{pare_result}', cam_dir=f'{cam_dir}', visualize=False,
                         model_folder=f'{MODEL_FOLDER}', vposer_ckpt=f'{VPOSER_FOLDER}',
                         part_segm_fn='/root/code/mover/smplifyx-file/smplx_parts_segm.pkl',gender='male', check_inverse_feet=False)
    print('********************************************')
    args2 = parser2.parse_args()
    args_dict2 = vars(args2)
    print(args_dict2)


    main0(args1)
    main1(**args2)
    # args2= args_pare_result()


    # print(args1.output_folder)

# args=vars(args)
# print(args)

# def f(**args):
#     print(args.output_folder)
#
# f(**args)
# main0(args)
