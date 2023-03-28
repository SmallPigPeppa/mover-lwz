import torch
from pare.models.layers import interpolate
from pare.models.head.pare_head import PareHead

head = PareHead(24, 2048)

heatmaps = torch.rand(1, 3, 56, 56, 56)
heatmaps[0, 0, 16, 17, 24] = 100.
keypoints, _ = head._softargmax3d(heatmaps, normalize_keypoints=True)

samples = interpolate(heatmaps, keypoints)
print(keypoints.shape)
print(keypoints)
print(samples)


# Sanity check of indexing
heatmaps = torch.zeros(1, 1, 7, 7)
heatmaps[0, 0, 1, 3] = 100.
keypoints, _ = head._softargmax2d(heatmaps, normalize_keypoints=True)

samples = interpolate(heatmaps, keypoints)
print(samples[0,0])

print(heatmaps)