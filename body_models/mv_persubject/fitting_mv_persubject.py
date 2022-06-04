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


# ! single image smplify-x
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from smplifyx.mesh_viewer import MeshViewer
from smplifyx import utils_mics
from smplifyx import misc_utils
from smplifyx import fitting as single_view_fitting
from loguru import logger 

from ..constants import (
    DEFAULT_LOSS_WEIGHTS,
)

class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=30, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])
    
    def close_viewer(self, ):
        if self.visualize:
            self.mv.close_viewer()
    
    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)
            # prev_loss = loss.item()

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = misc_utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, cameras=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, cameras=cameras,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return single_view_fitting.SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return single_view_fitting.SMPLifyCameraInitLoss(**kwargs)
    elif loss_type == 'body_orient':
        return SMPLifyBodyOrientLoss(**kwargs)
    elif loss_type == 'body_orient_multiview':
        return MultiViewSMPLifyBodyOrientLoss(**kwargs)
    elif loss_type == 'multiview_smplify':
        return MultiViewSMPLifyLoss(**kwargs)
    elif loss_type == '3D_joint_loss':
        return SMPLifyLoss3D(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))

class SMPLifyLoss3D(single_view_fitting.SMPLifyLoss):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss3D, self).__init__(search_tree=search_tree,
                 pen_distance=pen_distance, tri_filtering_module=tri_filtering_module,
                 rho=rho,
                 body_pose_prior=body_pose_prior,
                 shape_prior=shape_prior,
                 angle_prior=angle_prior,
                 dtype=torch.float32,
                 data_weight=data_weight,
                 body_pose_weight=body_pose_weight,
                 shape_weight=shape_weight,
                 bending_prior_weight=bending_prior_weight,
                 reduction='sum',
                 **kwargs)
        # import pdb;pdb.set_trace()
        self.ground_plane_support = kwargs['ground_plane_support']
        self.gp_support_loss_weight = kwargs['gp_support_loss_weight_init']

    def forward(self, body_model_output, gt_joints,
                body_model_faces,
                use_vposer=False, pose_embedding=None, 
                gt_ground_plane=None,
                **kwargs):
        # projected_joints = camera(body_model_output.joints)
        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        import pdb;pdb.set_trace()
        logger.info('run 3D joint loss.')
        joint_len = gt_joints.shape[1]
        print('joints shape', gt_joints.shape[-1])
        if gt_joints.shape[-1] == 3:
            joint_diff = (gt_joints - body_model_output.joints[:,:joint_len,:]) ** 2
        elif gt_joints.shape[-1] == 4:
            joint_diff = (gt_joints[:, :, :-1] - body_model_output.joints[:,:joint_len,:]) ** 2
            joint_diff *= gt_joints[:, :, -1]

        joint_loss = (torch.sum(joint_diff) * self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight
        total_loss = (joint_loss + pprior_loss + shape_loss + angle_prior_loss)

        message = f'Init joint: {joint_loss},  \
            angle_prior: {angle_prior_loss}'

        if self.ground_plane_support:    
            ## ground plane support loss
            assert gt_ground_plane is not None
            gp_support_loss = F.smooth_l1_loss(torch.max(body_model_output.vertices[:, :, 1]), gt_ground_plane)
            gp_support_loss = gp_support_loss * self.gp_support_loss_weight
            total_loss = total_loss + gp_support_loss
            message += f'gp: {gp_support_loss}'
        
        logger.info(message)

        return total_loss

class MultiViewSMPLifyLoss(single_view_fitting.SMPLifyLoss):
    def __init__(self, **kwargs):

        super(MultiViewSMPLifyLoss, self).__init__(**kwargs)
        ## ground_plane_support
        self.ground_plane_support = kwargs['ground_plane_support']
        if self.ground_plane_support:
            self.register_buffer('gp_support_loss_weight',
                                    torch.tensor(kwargs['gp_support_loss_weight'], dtype=kwargs['dtype']))
        
        ## ground_contact_support
        self.ground_contact_support = kwargs['ground_contact_support']
        if self.ground_contact_support:
            self.register_buffer('gp_contact_loss_weight',
                                    torch.tensor(kwargs['gp_contact_loss_weight'], dtype=kwargs['dtype']))
            self.ground_contact_vertices_ids = kwargs['ground_contact_vertices_ids']
        

        ## human & scene object inter-penetration 
        self.sdf_penetration = kwargs['sdf_penetration']
        if self.sdf_penetration:
            self.register_buffer('sdf_penetration_loss_weight',
                                    torch.tensor(kwargs['sdf_penetration_loss_weight'], dtype=kwargs['dtype']))
        
        ## human & scene object contact loss
        self.contact = kwargs['contact']
        if self.contact:
            self.register_buffer('contact_loss_weight',
                                    torch.tensor(kwargs['contact_loss_weight'], dtype=kwargs['dtype']))
            self.contact_verts_ids = kwargs['contact_verts_ids']
            self.rho_contact = kwargs['rho_contact']
            self.contact_angle = kwargs['contact_angle']
            self.contact_robustifier = misc_utils.GMoF_unscaled(rho=self.rho_contact)

        self.scene = kwargs['scene']
        if self.scene:
            self.register_buffer('scene_loss_weight',
                                    torch.tensor(kwargs['scene_loss_weight'], dtype=kwargs['dtype']))
        
        ## pare pose loss weight
        self.pare_pose_prior = kwargs['pare_pose_prior']
        if self.pare_pose_prior:
            self.register_buffer('pare_pose_weight', 
                                torch.tensor(kwargs['pare_pose_weight'], dtype=kwargs['dtype']))
        # import pdb;pdb.set_trace()

    def forward(self, body_model_output, cameras, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                gt_ground_plane=None, gt_contact_value=None, 
                scene_model=None, ftov=None,
                pare_body_pose=None,
                ## contact input
                **kwargs):

        joint_loss = 0
        for vi, cam in enumerate(cameras):
            projected_joints = cam(body_model_output.joints)
            # Calculate the weights for each joints
            weights = (joint_weights * joints_conf[vi, :]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)

            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            joint_diff = self.robustifier(gt_joints[vi, :, :, :] - projected_joints)
            # joint_diff = (gt_joints[vi, :, :, :] - projected_joints) ** 2
            joint_loss += (torch.sum(weights ** 2 * joint_diff) *
                        self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
        
        # TODO: pare pose prior on body pose
        pare_pose_prior_loss = 0.0
        #TODO:  if pare_pose not exists: Need to modifed.
        if self.pare_pose_prior and pare_body_pose is not None:
            import pdb;pdb.set_trace()
            # pose embedding to body pose (angles)
            pare_pose_prior_loss = torch.sum((1 - joints_conf[0,:]) * torch.abs(body_pose - pare_body_pose).pow(2))
            pare_pose_prior_loss = pare_pose_prior_loss * self.pare_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            
            batch_size = gt_joints.shape[1]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))
        
        gp_support_loss = 0.0
        if self.ground_plane_support:    
            ## ground plane support loss
            # import pdb;pdb.set_trace()
            assert gt_ground_plane is not None
            # if self.gp_support_loss_weight > 0.0:
            #     print('gp loss > 0 is {}'.format( self.gp_support_loss_weight))
            gp_support_loss = F.smooth_l1_loss(torch.max(body_model_output.vertices[:, :, 1]), gt_ground_plane)
            gp_support_loss = gp_support_loss * self.gp_support_loss_weight
        
        gp_contact_loss = 0.0
        if self.ground_contact_support:
            # TODO:     
            assert gt_contact_value is not None
            # if self.gp_contact_loss_weight > 0.1:
            #   import pdb; pdb.set_trace()   
            # the distance from heel and toe to the mesh surface should be 
            # consist with the distance from joint to the ground plane
            
            gt_contact_value[gt_contact_value==0] = -0.2
            gp_contact_loss = torch.abs(body_model_output.vertices[0, self.ground_contact_vertices_ids, 1] - gt_ground_plane)
            gp_contact_loss = (gt_contact_value * gp_contact_loss.mean(1) * self.gp_contact_loss_weight).mean()

        sdf_penetration_loss = 0.0
        if self.sdf_penetration and self.sdf_penetration_loss_weight >0:
            assert scene_model is not None
            sdf_penetration_loss = scene_model(smplx_model=body_model_output, stage=1)
            sdf_penetration_loss = self.sdf_penetration_loss_weight * sdf_penetration_loss
        
        # Compute the contact loss
        contact_loss = 0.0
        if self.contact and self.contact_loss_weight >0:
            # import pdb;pdb.set_trace()
            assert scene_model is not None
            contact_loss = scene_model(body_model_output, stage=2, contact_verts_ids=self.contact_verts_ids,
                                         contact_angle=self.contact_angle, contact_robustifier=self.contact_robustifier, ftov=ftov)
            contact_loss = self.contact_loss_weight * contact_loss
        # import pdb;pdb.set_trace()
        scene_loss = 0.0
        if self.scene and self.scene_loss_weight > 0:
            assert scene_model  is not None
            init_pid = 0
            scene_local_loss_weights = DEFAULT_LOSS_WEIGHTS[f'stage0_init{init_pid}']['loss_weight']
            scene_loss_tmp = scene_model(stage=0, loss_weights=scene_local_loss_weights)
            scene_loss_dict_weighted = {
                    k: scene_loss_tmp[k] * scene_local_loss_weights[k.replace("loss", "lw")] for k in scene_loss_tmp
                }
            
            scene_loss = sum(scene_loss_dict_weighted.values()).sum()
            scene_loss = self.scene_loss_weight * scene_loss
            
        # TODO: experiments to analysis;
        message = f'ST2 joint: {joint_loss}, \
            angle_prior: {angle_prior_loss}, pare_pose_prior_loss: {pare_pose_prior_loss}, pen:{pen_loss}, gp: {gp_support_loss}, \
            gp_contact: {gp_contact_loss}, sdf: {sdf_penetration_loss}, contact: {contact_loss}, scene: {scene_loss}'
        logger.info(message)
        
        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss
                      + gp_support_loss + gp_contact_loss + 
                      sdf_penetration_loss + contact_loss + pare_pose_prior_loss
                      + scene_loss)
        
        return total_loss

class SMPLifyBodyOrientLoss(nn.Module):

    def __init__(self, init_joints_idxs,
                 reduction='sum',
                 data_weight=1.0, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyBodyOrientLoss, self).__init__()
        self.dtype = dtype        

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            misc_utils.to_tensor(init_joints_idxs, dtype=torch.long))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2
       
        return joint_loss

class MultiViewSMPLifyBodyOrientLoss(SMPLifyBodyOrientLoss):

    def __init__(self, init_joints_idxs,
                 reduction='sum',
                 data_weight=1.0, dtype=torch.float32,
                 **kwargs):
        super(MultiViewSMPLifyBodyOrientLoss, self).__init__(init_joints_idxs=init_joints_idxs, data_weight=data_weight)

    def forward(self, body_model_output, cameras, gt_joints,
                **kwargs):
        
        joint_loss = 0
        for v_id, cam in enumerate(cameras):
            projected_joints = cam(body_model_output.joints)
            joint_error = torch.pow(
                torch.index_select(gt_joints[v_id, :, :, :], 1, self.init_joints_idxs) -
                torch.index_select(projected_joints, 1, self.init_joints_idxs),
                2)
            joint_loss += torch.sum(joint_error) * self.data_weight ** 2
       
        return joint_loss



###################################
#
# not fully tested temporal energy terms
#
###################################
class MultiViewTempSMPLifyLoss(MultiViewSMPLifyLoss):
    def __init__(self, pose_embedding_t_1, pose_embedding_t_2 = None, 
                temporal_smooth_weight=100.0, dtype=torch.float32, **kwargs):

        super(MultiViewTempSMPLifyLoss, self).__init__(**kwargs)

        self.register_buffer(
            'pose_embedding_t_1',torch.tensor(pose_embedding_t_1, dtype=dtype))

        if pose_embedding_t_2 is None:
            self.register_buffer(
                'pose_embedding_t_2', None)
        else:
            self.register_buffer(
                'pose_embedding_t_2',torch.tensor(pose_embedding_t_1, dtype=dtype))
        
        self.register_buffer(
            'temporal_smooth_weight',
            torch.tensor(temporal_smooth_weight, dtype=dtype))

    def forward(self, body_model_output, cameras, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):

        frame_wise_loss = super(MultiViewTempSMPLifyLoss, self).forward(body_model_output, 
                                                                        cameras=cameras,
                                                                        gt_joints=gt_joints,
                                                                        body_model_faces=body_model_faces,
                                                                        joints_conf=joints_conf,
                                                                        joint_weights=joint_weights,
                                                                        pose_embedding=pose_embedding,
                                                                        use_vposer=use_vposer,
                                                                        **kwargs)
        
        if self.pose_embedding_t_2 is not None:
            ##
            ## const speed smoothness term, not implemented yet
            ##

            # temporal_loss = (torch.sum( ((body_model_output.joints[:,:67,:] + 
            #                                 self.pose_embedding_t_2[:,:67,:]) /2 
            #                                 - self.pose_embedding_t_1[:,:67,:])**2 ) * 
            #                         self.temporal_smooth_weight ** 2)

            raise ValueError('Const speed smoothness term not implemented yet')
        elif self.pose_embedding_t_2 is None and self.pose_embedding_t_1 is not None:

            # temporal_loss = (torch.sum( (body_model_output.joints[:,:67,:] - 
            #                             self.pose_embedding_t_1[:,:67,:])**2 ) * 
            #                         self.temporal_smooth_weight ** 2)

            temporal_loss = (torch.sum( (pose_embedding - 
                                        self.pose_embedding_t_1)**2 ) * 
                                    self.temporal_smooth_weight ** 2)
        else:
            raise ValueError('No prev. results specified')
        # print(temporal_loss)
        
        total_loss = frame_wise_loss + temporal_loss
        return total_loss
