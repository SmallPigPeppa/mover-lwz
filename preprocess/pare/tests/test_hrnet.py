import torch

from pare.models.backbone.hrnet_pare import hrnet_w32, hrnet_w48

inp = torch.rand(1,3,224,224)
m = hrnet_w32(pretrained=False, pretrained_ckpt=False)

o = m(inp)
print(o.shape)

# import IPython; IPython.embed(); exit()