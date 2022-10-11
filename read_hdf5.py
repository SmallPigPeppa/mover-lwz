import h5py
filename = "samples_clean/3_3_78_Female2_0.hdf5"

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())