import open3d as o3d
import trimesh
import sys
# m = o3d.io.read_triangle_mesh('logs/occlusion_meshes/spin_ckpt/headtop.ply')
# o3d.visualization.draw_geometries([m])

m = trimesh.load(sys.argv[1])
m.show()