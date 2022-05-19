import sys
import json
import numpy as np

acc_results = json.load(open(sys.argv[1]))
acc_arr = []
for acc in acc_results:
    if acc[1]['val_mpjpe'] < 10: continue
    acc_arr.append([acc[1]['val_mpjpe'], acc[1]['val_pampjpe']])

accuracy = np.array(acc_arr)

for i in range(accuracy.shape[0]):
    print(i, accuracy[i])

print(f'Best MPJPE is {accuracy[:,0].min()} '
      f'at Epoch {accuracy[:,0].argmin()}, '
      f'result {accuracy[accuracy[:,0].argmin()]}')


print(f'Best PA-MPJPE is {accuracy[:,1].min()} '
      f'at Epoch {accuracy[:,1].argmin()}, '
      f'result {accuracy[accuracy[:,1].argmin()]}')