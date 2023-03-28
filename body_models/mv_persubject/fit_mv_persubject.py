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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict
import copy
import cv2
import PIL.Image as pil_img

from smplifyx.optimizers import optim_factory

import fitting_mv_persubject as fitting
from human_body_prior.tools.model_loader import load_vposer
import json
from psbody.mesh import Mesh
import scipy.sparse as sparse
from loguru import logger
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# if 'GPU_DEVICE_ORDINAL' in os.environ:
#     os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
# import pyrender

def fit_multi_view(img,
                keypoints,
                body_model,
                cameras,
                initialization,
                joint_weights,
                body_pose_prior,
                jaw_prior,
                left_hand_prior,
                right_hand_prior,
                shape_prior,
                expr_prior,
                angle_prior,
                result_fn='out.pkl',
                mesh_fn='out.obj',
                out_img_fn='overlay.png',
                loss_type='multiview_smplify',
                use_cuda=True,
                init_joints_idxs=(9, 12, 2, 5),
                use_face=True,
                use_hands=True,
                data_weights=None,
                body_pose_prior_weights=None,
                hand_pose_prior_weights=None,
                jaw_pose_prior_weights=None,
                shape_weights=None,
                expr_weights=None,
                hand_joints_weights=None,
                face_joints_weights=None,
                depth_loss_weight=1e2,
                interpenetration=True,
                coll_loss_weights=None,
                df_cone_height=0.5,
                penalize_outside=True,
                max_collisions=8,
                point2plane=False,
                part_segm_fn='',
                side_view_thsh=25.,
                rho=100,
                vposer_latent_dim=32,
                vposer_ckpt='',
                use_joints_conf=False,
                interactive=True,
                visualize=False,
                save_meshes=True,
                degrees=None,
                batch_size=1,
                dtype=torch.float32,
                ign_part_pairs=None,
                left_shoulder_idx=2,
                right_shoulder_idx=5,
                start_opt_stage=0,
                ####################
                ### PROX-POSA EXTENSION
                scene_model=None,
                update_scene=False,
                render_results=True,
                ## Camera
                camera_mode='moving',
                ## Groud Support Loss
                ground_plane_support=False,
                ground_plane_value=None,
                gp_support_weights_init=0.0,
                gp_support_weights=None,
                ground_contact_support=False,
                ground_contact_value=None,
                gp_contact_weights=None,
                #penetration
                sdf_penetration=False,
                sdf_penetration_loss_weight=None,
                sdf_dir=None,
                cam2world_dir=None,
                #contact
                contact=False,
                rho_contact=1.0,
                contact_loss_weights=None,
                contact_angle=15,
                contact_body_parts=None,
                body_segments_dir=None,
                load_scene=False,
                scene_dir=None,
                scene=False,
                scene_loss_weight=None,
                ## pare pose prior
                pare_pose_prior=False,
                pare_pose_weight=None,
                ## Depth
                s2m=False,
                s2m_weights=None,
                m2s=False,
                m2s_weights=None,
                rho_s2m=1,
                rho_m2s=1,
                init_mode=None,
                trans_opt_stages=None,
                viz_mode='mv',
                ## video smooth
                **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    assert 'transl' in initialization, 'We need to have global translation before fitting'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    beta_precomputed = kwargs.get('beta_precomputed', False)
    if beta_precomputed:
        beta_path = kwargs.get('beta_path',None)
        if beta_path:
            # import pdb;pdb.set_trace()
            with open(beta_path,'rb') as pkl_f:
                # betas = pickle.load(pkl_f) # TODO: joblib and pickle is different
                import joblib
                betas = joblib.load(pkl_f)['betas']
            betas_num = body_model.betas.shape[1]

            body_model.reset_params(betas=torch.tensor(betas[:betas_num], device=device))
            body_model.betas.requires_grad = False  # betas provided externally, not optimized
        else:
            print('beta_precomputed == True but no beta files (.pkl) found.')
            exit()

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)


    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    ## Ground Plane Support Loss
    if ground_plane_support:
        opt_weights_dict['gp_support_loss_weight'] = gp_support_weights
    if ground_contact_support:
        opt_weights_dict['gp_contact_loss_weight'] = gp_contact_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration: ## Human && Objs inter-penetration loss
        opt_weights_dict['sdf_penetration_loss_weight'] = sdf_penetration_loss_weight
    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights
    if pare_pose_prior:
        opt_weights_dict['pare_pose_weight'] = pare_pose_weight
    import pdb;pdb.set_trace()
    # TODO: add loss at five place: cfg, cmd_parser, input_def and in fit_mv, and in loss definition
    if scene:
        opt_weights_dict['scene_loss_weight'] = scene_loss_weight

    ## Grond Contact Loss
    ground_contact_vertices_ids = None
    if ground_contact_support: 
        ground_contact_vertices_ids = []
        for part in [ 'L_feet_front', 'L_feet_back', 'R_feet_front', 'R_feet_back']:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                ground_contact_vertices_ids.append(list(set(data["verts_ind"])))
        ground_contact_vertices_ids = np.stack(ground_contact_vertices_ids)

    contact_vertices_ids = ftov = None
    if contact:
        contact_verts_ids = []
        for part in contact_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
        contact_verts_ids = np.concatenate(contact_verts_ids)

        vertices = body_model(return_verts=True, body_pose= torch.zeros((batch_size, 63), dtype=dtype, device=device)).vertices
        vertices_np = vertices.detach().cpu().numpy().squeeze()
        body_faces_np = body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3)
        m = Mesh(v=vertices_np, f=body_faces_np)
        ftov = m.faces_by_vertex(as_sparse_matrix=True)

        ftov = sparse.coo_matrix(ftov)
        indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
        values = torch.FloatTensor(ftov.data).to(device)
        shape = ftov.shape
        ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))


    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # keypoints = [torch.tensor(k, dtype=dtype) for k in keypoints]
    keypoint_data = torch.stack(keypoints)
    gt_joints = keypoint_data[:, :, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, :, 2].reshape(len(keypoints), -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)
    body_orientation_loss = fitting.create_loss('body_orient_multiview',
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)

    # set scene parameters
    # if not update_scene:
    #     for param in scene_model.parameters():
    #         print(f'set params in scene from {param.requires_grad} to {False}')
    #         param.requires_grad = False
    # else:
    #     # import pdb;pdb.set_trace()
    #     update_list = ['rotate_cam_roll', 'rotate_cam_pitch', 'rotations_object', 'translations_object']
    #     for key, param in scene_model.named_parameters():
    #         if key in update_list:
    #             print(f'set {key} grad: True')
    #             param.requires_grad = True
    
    # define loss
    # import pdb;pdb.set_trace()
    init_loss = fitting.create_loss("3D_joint_loss",
                               joint_weights=joint_weights,
                               rho=rho,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               data_weight=1e1,
                               body_pose_weight=1.0,
                               shape_weight=1e2,
                               dtype=dtype,
                               ground_plane_support=ground_plane_support,
                               gp_support_loss_weight_init=gp_support_weights_init, #1e1
                               ).to(device=device)
   
    loss = fitting.create_loss("multiview_smplify",
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               ##HDSR
                               ground_plane_support=ground_plane_support,
                               ground_contact_support=ground_contact_support,
                               ground_contact_vertices_ids=ground_contact_vertices_ids,
                               gp_support_loss_weight=gp_support_weights, # init sample, only one, will be replaced during optimization
                               gp_contact_loss_weight=gp_contact_weights,
                               sdf_penetration=sdf_penetration,
                               scene_model=scene_model,
                               sdf_penetration_loss_weight=sdf_penetration_loss_weight,
                               ## contact loss, TODO: change to POSA
                               contact=contact,
                               contact_verts_ids=contact_verts_ids,
                               rho_contact=rho_contact,
                               contact_angle=contact_angle,
                               contact_loss_weight=contact_loss_weights,
                               ## pare pose prior
                               pare_pose_prior=pare_pose_prior,
                               pare_pose_weight=pare_pose_weight,
                               ## scene loss:
                               scene=scene,
                               scene_loss_weight=scene_loss_weight,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:        

        H, _, _ = torch.tensor(img[0], dtype=dtype).shape

        data_weight = 1000 / H


         # Step 1: Initialization
         # Two options:
         # 1. Optimize the full pose using 3D skl provided externally (OpenPose or MvPose)
         # 2. Optimize the body orientation using the torso joints
            
        if start_opt_stage > 0:     # if the starting stage bigger than 0: fit to 3D skeletons

            # Reset the parameters to mean pose
            # TODO: kpts_3d is wrong, fixed on 17.01
            new_kpts_3d = initialization['keypoints_3d'][:15, :3]
            new_kpts_3d[:, 1] = -1 * new_kpts_3d[:, 1] 
            gt_joints_3d = torch.tensor(new_kpts_3d, device=device)
            # gt_joints_3d = torch.tensor(initialization['keypoints_3d'][:15, :3], device=device) 
            gt_joints_3d.unsqueeze_(0)

            if beta_precomputed:
                body_model.reset_params(body_pose=body_mean_pose, betas=torch.tensor(betas[:betas_num], device=device))
            else:
                body_model.reset_params(body_pose=body_mean_pose)

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            init_params = [body_model.global_orient, body_model.transl]
            if use_vposer:
                init_params.append(pose_embedding)

            init_optimizer, init_create_graph = optim_factory.create_optimizer(
                                                    init_params,
                                                    **kwargs)
            init_optimizer.zero_grad()

            import pdb;pdb.set_trace()
            # build loss closure
            fit_init = monitor.create_fitting_closure(
                                    init_optimizer, body_model, cameras=None,
                                    loss=init_loss, gt_joints=gt_joints_3d, create_graph=init_create_graph,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding, gt_ground_plane=ground_plane_value ,
                                    return_full_pose=True, return_verts=True)
            # fitting process
            ori_init_start = time.time()
            init_loss_val = monitor.run_fitting(init_optimizer,
                                                    fit_init,
                                                    init_params, body_model, 
                                                    gt_joints=gt_joints_3d,
                                                    use_vposer=use_vposer,
                                                    pose_embedding=pose_embedding,
                                                    vposer=vposer,
                                                    gt_ground_plane=ground_plane_value)
        
        else:   # if the starting stage == 0: find global orientation first
            # The closure passed to the optimizer
            body_orientation_loss.reset_loss_weights({'data_weight': data_weight})

            # Reset the parameters to estimate the initial translation of the
            # body model
            if beta_precomputed:
                body_model.reset_params(body_pose=body_mean_pose,
                                        transl=initialization['transl'], #                                           scale=1.0, 
                                        betas=torch.tensor(betas[:betas_num], device=device))
            else:
                body_model.reset_params(body_pose=body_mean_pose, transl=initialization['transl'])

            body_orientation_opt_params = [body_model.global_orient, body_model.transl]

            body_orientation_optimizer, body_orientation_create_graph = optim_factory.create_optimizer(
                body_orientation_opt_params,
                **kwargs)

            # The closure passed to the optimizer
            fit_camera = monitor.create_fitting_closure(
                                    body_orientation_optimizer, body_model, cameras, gt_joints,
                                    body_orientation_loss, create_graph=body_orientation_create_graph,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding, gt_ground_plane=ground_plane_value,
                                    return_full_pose=False, return_verts=False)

            # Initialize the computational graph by feeding the initial translation
            # of the camera and the initial pose of the body model.
            ori_init_start = time.time()
            init_loss_val = monitor.run_fitting(body_orientation_optimizer,
                                                    fit_camera,
                                                    body_orientation_opt_params, body_model,
                                                    use_vposer=use_vposer,
                                                    pose_embedding=pose_embedding,
                                                    vposer=vposer,
                                                    gt_ground_plane=ground_plane_value)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            if start_opt_stage > 0:
                tqdm.write('Initialized with 3D skl done after {:.4f} sec.'.format(
                    time.time() - ori_init_start))
                tqdm.write('Initialized with 3D skl final loss {:.4f}'.format(
                    init_loss_val))

            else:
                tqdm.write('Body orientation initialization done after {:.4f} sec.'.format(
                    time.time() - ori_init_start))
                tqdm.write('Body orientation initialization final loss {:.4f}'.format(
                    init_loss_val))

       
        orientations = [body_model.global_orient.detach().cpu().numpy()]        
        
        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()
            if start_opt_stage == 0:
                new_params = defaultdict(global_orient=orient,
                                        body_pose=body_mean_pose, 
                                        transl=body_model.transl, #    scale=1.0   # translations are in meters
                                        )
                if beta_precomputed:
                    new_params['betas'] = torch.tensor(betas[:betas_num], device=device)

                body_model.reset_params(**new_params)

                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)

            # stage: 2:6
            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[start_opt_stage:], desc='Stage')):
                # import pdb;pdb.set_trace()
                # Warning: Not update the scene, we could also generate a plausible human body
                # set update scene module
                if update_scene and opt_idx == len(opt_weights[start_opt_stage:]) - 2:
                    scene_model.set_active_scene(activate_list=['rotate_cam_pitch', 'rotate_cam_roll'])
                elif update_scene and opt_idx == len(opt_weights[start_opt_stage:]) - 1:
                    scene_model.set_active_scene(activate_list=['rotate_cam_pitch', 'rotate_cam_roll', 'rotations_object', 'translations_object'])
                else:
                    # set static scene
                    scene_model.set_static_scene()
                
                # fixed body parameters
                if False and opt_idx >= len(opt_weights[start_opt_stage:]) - 2:
                    # import pdb;pdb.set_trace()
                    for key, value in body_model.named_parameters():
                        if value.requires_grad == True:
                            logger.info(f'stage {opt_idx} set {key} False')
                            value.requires_grad = False
                        else:
                            logger.info(f'stage {opt_idx}  {key}  is False')
                    # import pdb;pdb.set_trace()
                    
                # break
                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)
                
                # TODO: use the same optimizer
                if update_scene:
                    scene_params = list(scene_model.parameters())
                    final_scene_params = list(
                        filter(lambda x: x.requires_grad, scene_params))
                    for one in final_scene_params:
                        final_params.append(one)
                    # scene_optimizer, scene_create_graph = optim_factory.create_optimizer(
                    #     final_scene_params,     **kwargs)
                    # scene_optimizer.zero_grad()

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()
                
                # TODO: set weight for each stage
                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)
                
                # TODO:  most important, define loss
                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    cameras=cameras, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding, 
                    gt_ground_plane=ground_plane_value, gt_contact_value=ground_contact_value,
                    scene_model=scene_model,
                    ftov=ftov,
                    # TODO: add scene contact
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                # TODO: add tensorboard monitor
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer,
                    # TODO: add scene contact
                    scene_model=scene_model)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d}/{} done after {:.4f} seconds'.format(
                            opt_idx+start_opt_stage, len(opt_weights), elapsed))

            # TODO: fix bug in Visulization 
            monitor.close_viewer()
            # print("{} -> {}".format(initialization['transl'], body_model.transl))
            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {key: val.detach().cpu().numpy()
                            for key, val in body_model.named_parameters()}
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        
        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0

            body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1) if use_vposer else None

            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            model_output = body_model(return_verts=True, body_pose=body_pose)
            results[min_idx]['result']['keypoints_3d'] = model_output.joints.detach().cpu().numpy().squeeze()[:25, :]
            results[min_idx]['result']['body_pose'] = body_pose.detach().cpu().numpy()
            results[min_idx]['result']['pose_embedding'] = pose_embedding.detach().cpu().numpy()
            results[min_idx]['result']['gender'] = kwargs['gender']
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)


        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        import trimesh
        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        
        if save_meshes or visualize:
            out_mesh.export(mesh_fn)
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
-           if 'GPU_DEVICE_ORDINAL' in os.environ:
-               os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
-           import pyrender
            for ci, cam in enumerate(tqdm(cameras, desc="Rendering overlays")):

                # scene
                material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            wireframe=True,
                            roughnessFactor=.5,
                            alphaMode='OPAQUE',
                            baseColorFactor=(0.9, 0.5, 0.9, 1))


                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

                # mesh
                rot_mat = cam.rotation.detach().cpu().numpy().squeeze()
                translation = cam.translation.detach().cpu().numpy().squeeze()
                out_mesh.vertices = np.matmul(vertices, rot_mat.T) + translation

                mesh = pyrender.Mesh.from_trimesh(
                    out_mesh,
                    material=material)

                
                scene.add(mesh, 'mesh')

                # img
                input_img = img[ci]
                height, width = input_img.shape[:2]
                
                center = cam.center.detach().cpu().numpy().squeeze().tolist()

                camera_pose = np.eye(4)
                # camera_pose = RT
                camera_pose[1, :] = - camera_pose[1, :]
                camera_pose[2, :] = - camera_pose[2, :]

                camera = pyrender.camera.IntrinsicsCamera(
                    fx=cam.focal_length_x.item(), fy=cam.focal_length_y.item(),
                    cx=center[0], cy=center[1])
                scene.add(camera, pose=camera_pose)

                # Get the lights from the viewer
                light_node = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                scene.add(light_node, pose=camera_pose)

                r = pyrender.OffscreenRenderer(viewport_width=width,
                                                viewport_height=height,
                                                point_size=1.0)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0

                valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

                output_img = (color[:, :, :-1] * valid_mask +
                                (1 - valid_mask) * input_img)

                output_img = pil_img.fromarray((output_img * 255.).astype(np.uint8))
                output_img.save("{}/../images/{:02}.png".format(kwargs['result_folder'], ci))

    return copy.deepcopy(out_mesh), copy.deepcopy(scene_model)
