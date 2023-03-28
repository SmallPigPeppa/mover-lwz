# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

from ..body_models.smplifyx.utils import JointMapper
from ..body_models.smplifyx.data_parser import create_dataset
from .fit_mv_persubject import fit_multi_view
from ..body_models.smplifyx.multiview_initializer import MultiViewInitializer
from ..body_models.smplifyx.camera import create_multicameras
from ..body_models.smplifyx.prior import create_prior

torch.backends.cudnn.enabled = False

def main_mv(scene_prior, **args):

    # output, save as list
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    # input 
    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    # model parameters
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=dtype,
                        **args)
    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # Only one camera info
    xml_folder = args.get('calib_path', None)
    if xml_folder is not None:    
        if xml_folder != '':
            cameras = create_multicameras(xml_folder=xml_folder,
                            dtype=dtype,
                            **args)
        else:
            raise ValueError('Path must be specified!')
        

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    # prior information
    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():args
        device = torch.device('cuda')
        for c in cameras:
            c.to(device=device)
            if hasattr(c, 'rotation'):
                c.rotation.requires_grad = False

        # cameras = cameras.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    # initialization
    # here it always uses the 3D skl from openpose
    start_opt_stage = args.pop('start_opt_stage', 0)
    initialization = MultiViewInitializer(dataset_obj, result_path=None, **args).get_init_params()
    
    # TODO: load input kpts and camera 
    img = []
    keypoints = []
    detectionTF = []
    for vi, data in enumerate(dataset_obj):
        if (data):
            if False:
                # TODO: warning if the image is 1/4 downsampled for Multi-IOI dataset
                import cv2
                h, w, c = data['img'].shape
                img.append(cv2.resize(data['img'], (4*w, 4*h)))
            else:
                img.append(data['img'])
            keypoints.append(torch.tensor(data['keypoints']))
            detectionTF.append(True)
        else:
            detectionTF.append(False)
    cameras = [c for ii, c in enumerate(cameras) if detectionTF[ii]]
    
    # per-joint view weight adjustment
    if len(cameras) > 1:
        from view_aposteriori import multiview_weight_adjustment
        keypoints = multiview_weight_adjustment(keypoints, cameras, args['sigma'])
    else:
        print('run single view version !!!!!!')

    fn = "."
    curr_result_folder = osp.join(result_folder, fn)
    if not osp.exists(curr_result_folder):
        os.makedirs(curr_result_folder)
    curr_mesh_folder = osp.join(mesh_folder, fn)
    if not osp.exists(curr_mesh_folder):
        os.makedirs(curr_mesh_folder)

    person_id = 0

    curr_result_fn = osp.join(curr_result_folder,
                                '{:03d}.pkl'.format(person_id))
    curr_mesh_fn = osp.join(curr_mesh_folder,
                            '{:03d}.obj'.format(person_id))

    curr_img_folder = osp.join(output_folder, 'images', fn)
    if not osp.exists(curr_img_folder):
        os.makedirs(curr_img_folder)

    if gender_lbl_type != 'none':
        if gender_lbl_type == 'pd' and 'gender_pd' in dataset_obj[0]:
            gender = dataset_obj[0]['gender_pd'][person_id]
        if gender_lbl_type == 'gt' and 'gender_gt' in dataset_obj[0]:
            gender = dataset_obj[0]['gender_gt'][person_id]
    else:
        gender = input_gender

    # TODO: set body model which contains parameters
    if gender == 'neutral':
        body_model = neutral_model
    elif gender == 'female':
        body_model = female_model
    elif gender == 'male':
        body_model = male_model
    args['gender'] = gender

    out_img_fn = osp.join(curr_img_folder, '{:03d}.png'.format(person_id))

    #TODO: update scene prior
    ground_plane_value = scene_prior['ground_plane']
    scene_model = scene_prior['scene_model']
    ground_contact_value = scene_prior['ground_contact_value'] # four labels and conf

    # TODO: load previous results and start from the last two stage
    
    # import pdb;pdb.set_trace()
    fitted_body_model, scene_model =fit_multi_view(img, keypoints,
                    body_model=body_model,
                    cameras=cameras,
                    initialization = initialization,
                    joint_weights=joint_weights,
                    dtype=dtype,
                    output_folder=output_folder,
                    result_folder=curr_result_folder,
                    out_img_fn=out_img_fn,
                    result_fn=curr_result_fn,
                    mesh_fn=curr_mesh_fn,
                    shape_prior=shape_prior,
                    expr_prior=expr_prior,
                    body_pose_prior=body_pose_prior,
                    left_hand_prior=left_hand_prior,
                    right_hand_prior=right_hand_prior,
                    jaw_prior=jaw_prior,
                    angle_prior=angle_prior,
                    start_opt_stage=start_opt_stage,
                    ground_plane_value=ground_plane_value,
                    scene_model=scene_model,
                    ground_contact_value=ground_contact_value,
                    **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

    return fitted_body_model, scene_model
