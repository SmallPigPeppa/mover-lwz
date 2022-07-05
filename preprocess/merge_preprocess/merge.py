import sys

sys.path.append('footskate_reducer')
sys.path.append('footskate_reducer/ground_detector')
from footskate_reducer.ground_detector.op_filter_json_merge import main0
from demo_merge import main1
from demo_pare_result_merge import main2
import argparse
from merge_parser import parser_pare, parser_pare_result, parser_op_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge preprocess for smplifyx-modified')
    parser.add_argument('--video_name', type=str, default='Color_flip', help='an integer for the accumulator')
    parser.add_argument('--out_dir', type=str, default='/root/code/mover/preprocess/out_data',
                        help='an integer for the accumulator')
    parser.add_argument('--root_dir', type=str, default='/root/code/mover/preprocess/input_data',
                        help='an integer for the accumulator')
    args = parser.parse_args()

    # video_name=args.video_name

    video_name = 'Color_flip'
    video_path = '/root/code/mover/preprocess/input_data/Color_flip/Color_flip.mp4'
    openpose_dir = '/root/code/mover/preprocess/input_data/Color_flip/openpose'
    image_dir = '/root/code/mover/preprocess/input_data/Color_flip/imgs'
    out_dir = '/root/code/mover/preprocess/out_data'
    # # step0: openpose filter
    # parser0 = parser_op_filter()
    # parser0.set_defaults(root=openpose_dir,
    #                     dump=f'{out_dir}/{video_name}/openpose_OneEurofilter',
    #                     img_dir=image_dir,
    #                     viz=True)
    # args0 = parser0.parse_args()
    # main0(args0)

    # # step1: pare
    # parser1 = parser_pare()
    # pare_model = '../pare/hrnet_model'
    pare_exp = 'pare'
    # parser1.set_defaults(cfg=f'{pare_model}/config.yaml', ckpt=f'{pare_model}/checkpoint.ckpt',
    #                      output_folder=f'{out_dir}/{video_name}', vid_file=f'{video_path}', draw_keypoints=True,
    #                      detector='maskrcnn',exp=pare_exp)
    # args1 = parser1.parse_args()
    # main1(args1)

    # step2: op2smplifyx_withPARE
    parser2 = parser_pare_result()
    output_folder = f'{out_dir}/{video_name}/op2smplifyx_withPARE'
    json_folder = f"{out_dir}/{video_name}/openpose_OneEurofilter"
    # pare_result = '/root/code/mover/preprocess/Color_flip/pare_output.pkl'
    pare_result = f'{out_dir}/{video_name}/{pare_exp}/pare_output.pkl'
    cam_dir = '/root/code/mover/smplifyx_cam'
    cam_dir = '../../smplifyx_cam'
    model_folder = 'data/body_models'
    vposer_folder = '../../smplifyx-file/vposer_v1_0'
    segm_fn_path = '../../smplifyx-file/smplx_parts_segm.pkl'
    '''
    #    --config ./cfg_files/fit_smpl.yaml \
    #    --export_mesh True \
    #    --save_new_json True \
    #    --json_folder  ${JSON_FOLDER} \
    #    --data_folder ${DATA_FOLDER} \
    #    --output_folder ${OUTPUT_FOLDER} \
    #    --pare_result ${pare_result} \
    #    --cam_dir ${cam_dir} \
    #    --visualize="False" \
    #    --model_folder ${MODEL_FOLDER} \
    #    --vposer_ckpt ${VPOSER_FOLDER} \
    #    --part_segm_fn /root/code/mover/smplifyx-file/smplx_parts_segm.pkl \
    #    --gender male \
    #    --check_inverse_feet="False" \
    '''

    parser2.set_defaults(config='cfg_files/fit_smpl.yaml', export_mesh=True, save_new_json=True,
                         json_folder=f'{json_folder}', data_folder=f'{image_dir}', output_folder=f'{output_folder}',
                         pare_result=f'{pare_result}', cam_dir=f'{cam_dir}', visualize=False,
                         model_folder=f'{model_folder}', vposer_ckpt=f'{vposer_folder}',
                         part_segm_fn=f'{segm_fn_path}', gender='male',
                         check_inverse_feet=False)
    args2 = parser2.parse_args()
    args_dict2 = vars(args2)
    print('********************************************')
    print(args_dict2)
    main2(**args_dict2)

    # print('********************************************')
    # print(args_dict2)
    # #
    # # print(args1)
    # # print('********************************************')
    # #
    # # args2 = parser2.parse_args()
    # # args_dict2 = vars(args2)
    # # print(args_dict2)
