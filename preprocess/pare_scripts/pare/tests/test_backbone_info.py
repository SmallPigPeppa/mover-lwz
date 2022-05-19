import torch
from pare.models.backbone import *
from loguru import logger

if __name__ == '__main__':
    inp = torch.rand(1,3,224,224)

    models = [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'resnext50_32x4d',
        'resnext101_32x8d',
        'wide_resnet50_2',
        'wide_resnet101_2',
        'mobilenet_v2',
        'hrnet_w32',
        'hrnet_w48',
        # 'dla34'
    ]

    for m in models:

        net = eval(m)()
        out = net(inp)
        logger.info(f'{m}: {out.shape}')
