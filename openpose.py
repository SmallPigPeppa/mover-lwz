##   This python script launches openpose to get 2D joints for all images under an input directory  ##
##   Author: Jinlong Yang
##   Email:  jinlong.yang@tuebingen.mpg.de

import os
import sys
from os.path import join, isdir, isfile
# from tqdm import tqdm
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/ps/project/common/singularity_openpose/examples')
parser.add_argument('--output_dir', type=str, default='/ps/project/common/singularity_openpose/output/')


def openpose_ini(openposepy_dir='/openpose/build/python/'):
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openposepy_dir);
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = join(openposepy_dir, "../../models/")
    params["hand"] = True
    params["face"] = True

    # # Add others in path?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1])-1: next_item = args[1][i+1]
    #     else: next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    return opWrapper, datum


def openpose_run(opWrapper, datum, image_file):
    # Process Image
    imageToProcess = cv2.imread(image_file)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    return datum


import json


def export_json(datum, file_name):
    dic = {}
    dic['version'] = '1.5'
    dic["people"] = []
    person_dic = {}
    if hasattr(datum, 'poseKeypoints'):
        if datum.poseKeypoints.shape:
            for i in range(0, datum.poseKeypoints.shape[0]):
                person_dic["person_id"] = [-1]
                person_dic["pose_keypoints_2d"] = (datum.poseKeypoints[0, :, :].reshape((-1))).tolist()
        else:
            person_dic["person_id"] = []
            person_dic["pose_keypoints_2d"] = []
            person_dic["pose_keypoints_2d"] = []

    if hasattr(datum, 'faceKeypoints'):
        if datum.faceKeypoints.shape:
            for i in range(0, datum.faceKeypoints.shape[0]):
                person_dic["face_keypoints_2d"] = (datum.faceKeypoints[0, :, :].reshape((-1))).tolist()
        else:
            person_dic["face_keypoints_2d"] = []

    if hasattr(datum, 'handKeypoints'):
        if datum.handKeypoints[0].shape:
            for i in range(0, datum.handKeypoints[0].shape[0]):
                person_dic["hand_left_keypoints_2d"] = (datum.handKeypoints[0][0, :, :].reshape((-1))).tolist()
        else:
            person_dic["hand_left_keypoints_2d"] = []

        if datum.handKeypoints[1].shape:
            for i in range(0, datum.handKeypoints[1].shape[0]):
                person_dic["hand_right_keypoints_2d"] = (datum.handKeypoints[1][0, :, :].reshape((-1))).tolist()
        else:
            person_dic["hand_right_keypoints_2d"] = []

    dic["people"].append(person_dic)

    with open(file_name, 'w') as fp:
        json.dump(dic, fp)
    return


def process_all_images_in_directory(input_dir, output_dir):  # input_dir contains .png or jpg images
    # Initialize  openpose wrapperh:
    opWrapper, datum = openpose_ini()

    if not isdir(output_dir):
        os.mkdir(output_dir)

    # Get all images under current directory
    images = sorted([s for s in os.listdir(input_dir) \
                     if (isfile(join(input_dir, s))) \
                     and (s[-3:] == 'jpg' or s[-3:] == 'png')])

    print(len(images), ' images in total will be processed.')
    # for image in tqdm(images):
    for image in images:
        print("Processing ", image)
        datum = openpose_run(opWrapper, datum, join(input_dir, image))
        poseKeypoints_2d = datum.poseKeypoints
        faceKeypoints_2d = datum.faceKeypoints
        handKeypoints_2d_left = datum.handKeypoints[0]
        handKeypoints_2d_right = datum.handKeypoints[1]

        json_file_name = join(output_dir, image[:-4] + '_keypoints.json')
        export_json(datum, json_file_name)

        # np.save(join(output_dir, image[:-4]+'_2djoint.npy'), poseKeypoints_2d)
        cv2.imwrite(join(output_dir, image[:-4] + '_openpose.png'), datum.cvOutputData)


if __name__ == '__main__':
    args = parser.parse_args()
    process_all_images_in_directory(args.input_dir, args.output_dir)

def test_openpose(input_dir, video_name, save_dir):
    shell = '/ps/scratch/multi-ioi/hyi/singularity_openpose/example_run_openpose_hyi.sub'
    # example_run_openpose_hyi.sub ？
    with open(shell, 'r') as fin:
        lines = fin.readlines()
    lines[1] = f'arguments = "exec --nv /ps/project/common/singularity_openpose/openpose.simg python3 \
        /ps/project/common/singularity_openpose/openpose_script.py \
        --input_dir {save_dir}/{video_name}/Color_flip_rename \
        --output_dir {save_dir}/{video_name}/Color_filp_rename_openpose"\n'

    with open(os.path.join(save_dir,video_name, 'process_shell', '3_test_openpose.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

'''
echo "Hello World !"
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand --write_json output_json_folder/
'''
