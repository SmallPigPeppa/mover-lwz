import h5py
import numpy as np
# oritation

temp_name = "samples_clean/3_3_78_Female2_0.hdf5"
gta_name = "samples_clean_gta/FPS-5-2020-06-11-10-06-48.hdf5"
info_npz_name = "samples_clean_gta/FPS-5/2020-06-11-10-06-48/info_frames.npz"

# info_npz = np.load(open(info_npz_name, 'rb'))
info_npz = np.load(info_npz_name)
print(info_npz['joints_3d_world'].shape)
with h5py.File(temp_name, "r+") as f:
    print("Keys: %s" % f.keys())
    del f['newarray']
    print("Keys: %s" % f.keys())
    f.create_dataset('skeleton_joints', data=info_npz['joints_3d_world'])
    print(f['skeleton_joints'])
    print("Keys: %s" % f.keys())


# with h5py.File(gta_name, "r") as f:
#     print("Keys: %s" % f.keys())
