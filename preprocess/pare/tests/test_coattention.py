import torch

from pare.models.layers.coattention import CoAttention

if __name__ == '__main__':
    inp1 = torch.rand(3, 256, 56, 56)
    inp2 = torch.rand(3, 256, 56, 56)

    for fc in ['double_1', 'double_3', 'single_1', 'single_3', 'simple']:
        c = CoAttention(n_channel=256, final_conv=fc)
        out1, out2 = c(inp1, inp2)
        assert inp1.shape == out1.shape
        assert inp2.shape == out2.shape
        print(fc, 'out1', out1.shape, 'out2', out2.shape)

