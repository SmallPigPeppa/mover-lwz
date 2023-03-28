import pickle
import os
data_root = os.path.join('samples_clean_gta', 'FPS-5')
rec_idx = '2020-06-11-10-06-48'
info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
kpts_pkl_names = [i['kpname'] for i in info_pkl]
# print(kpts_pkl_names[0])
# print(kpts_pkl_names[24])
for k in range(len(kpts_pkl_names)):
    for i in range(len(kpts_pkl_names[0])):
        if kpts_pkl_names[0][i]!=kpts_pkl_names[k][i]:
            print( kpts_pkl_names[0][i],kpts_pkl_names[k][i])