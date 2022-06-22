import skimage.io as io
import matplotlib.pyplot as plt

from pare.utils.vis_utils import colormap_to_arr

hm = io.imread('/home/mkocabas/Pictures/00.png')
print(hm.max())

val = colormap_to_arr(hm.reshape((-1, 3)))

print(val.shape)

val = val.reshape((hm.shape[0], hm.shape[1]))

plt.imshow(val)
plt.show()