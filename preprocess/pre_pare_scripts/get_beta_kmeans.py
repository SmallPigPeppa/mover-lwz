from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import glob
import os
import joblib
from tqdm import tqdm

pare_result_dir = '/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_result_pare3d_opfeetAnkles'
all_list = os.listdir(pare_result_dir)
X = []
tmp_result = None
for one in tqdm(all_list):
    # import pdb;pdb.set_trace()
    try:
        # print(f'process {one}')
        pare_fn = os.path.join(pare_result_dir, one, 'results/000.pkl')
        pare_result = joblib.load(pare_fn)
        X.append(pare_result['betas'])
        if tmp_result is None:
            tmp_result = pare_result
    except:
        print(f'ERROR {one}')
X = np.concatenate(X, 0)
all_num = X.shape[0]

dis = pairwise_distances(X[0:1,:], X[1:2,:])
print(f'pair dis: {dis}')

clustering = DBSCAN(eps=1.8, min_samples=int(all_num * 0.8)).fit(X)
n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"Number of clusters: {n_clusters_}")


print(clustering.labels_)

filter_label = [True if one == 0 else False for one in clustering.labels_ ]
filter_X = X[filter_label]
print(f'label : {np.array(filter_label).sum()}/{len(filter_label)}')
#np.clustering.labels_
kmeans = KMeans(n_clusters=1, random_state=0).fit(filter_X)
print('center:', kmeans.cluster_centers_)

# result = {}
result = tmp_result
result['betas'] = np.array(kmeans.cluster_centers_)
pickle_file = './mean_smplx.pkl'
from joblib import dump
with open(pickle_file, 'wb') as f:
    dump(result, f)



