import os
import time
import trimesh
import numpy as np
from os.path import join

from pare.utils.mesh_viewer import MeshViewer

def main(images_dir):
    obj_files = [join(images_dir,x) for x in os.listdir(images_dir) if x.endswith('.obj')]

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1,0,0])
    rot = trimesh.transformations.rotation_matrix(np.radians(45), [0,1,0]) @ rot

    mv = MeshViewer()
    for idx, obj_f in enumerate(obj_files):
        m = trimesh.load(obj_f)
        mv.update_mesh(m.vertices, m.faces, transform=rot)
        breakpoint()
        # m.show()


if __name__ == '__main__':
    images_dir = '/ps/scratch/mkocabas/developments/CVPR2020-OOH/output/demo/coco_qualitative_figures/images'
    main(images_dir)