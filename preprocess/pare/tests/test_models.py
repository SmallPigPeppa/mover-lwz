import torch
from thop import profile, clever_format

from pare.models import HMR, PARE
from pare.models.hmr_adf_dropout import HMR_ADF_DROPOUT


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    # model = HMR(backbone='mobilenet_v2')
    model = PARE(
        # backbone='hrnet_w48',
        backbone='mobilenet_v2',
        iterative_regression=False,
        iter_residual=False,
        num_iterations=1,
        shape_input_type='feats.shape.cam',
        pose_input_type='feats.self_pose',
        pose_mlp_num_layers=1,
        shape_mlp_num_layers=1,
        pose_mlp_hidden_size=256,
        shape_mlp_hidden_size=256,
        use_keypoint_features_for_smpl_regression=False,
        use_heatmaps='part_segm',
        use_keypoint_attention=True,
        use_postconv_keypoint_attention=True,
        keypoint_attention_act='sigmoid',
        use_scale_keypoint_attention=True,
        use_final_nonlocal='', # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
        use_branch_nonlocal='', # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
        use_hmr_regression=False,
        use_coattention=False,
        num_coattention_iter=1,
        coattention_conv='double_1',
        use_upsampling=True,
        deconv_conv_kernel_size=3,
        use_soft_attention=False,
        num_branch_iteration=0,
        branch_deeper=False,
        num_deconv_layers=3,
        num_deconv_filters=256,
        use_resnet_conv_hrnet=True,
        use_position_encodings=False,
    )
    # from pare.models.backbone.resnet_adf_dropout import test
    # test()
    # exit()

    # model = HMR(estimate_var=True)

    # activation = {}
    #
    #
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #
    #     return hook


    # model.head.keypoint_deconv_layers.register_forward_hook(get_activation('kp_conv'))

    # print(model)

    inp = torch.rand(1,3,224,224)

    output = model(inp)

    # act = activation['kp_conv'].squeeze()
    # breakpoint()

    macs, params = clever_format(profile(model, inputs=(inp,)), "%.3f")
    print('Macs', macs, 'Params', params)

    for k,v in output.items():
        print(k, v.shape)