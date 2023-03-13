import h5py
filename = "samples_clean_gta/FPS-5-clean-debug/FPS-5-2020-06-11-10-06-48.hdf5"

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    print(f.value())