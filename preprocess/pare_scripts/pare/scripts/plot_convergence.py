import matplotlib.pyplot as plt
import json
import numpy as np

def read_acc_results(f):
    acc_results = json.load(open(f))
    acc_arr = []
    for acc in acc_results:
        if acc[1]['val_mpjpe'] < 10: continue
        acc_arr.append([acc[1]['val_mpjpe'], acc[1]['val_pampjpe']])

    accuracy = np.array(acc_arr)
    return accuracy


acc_spin = read_acc_results('logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/val_accuracy_results.json')[:50]
acc_pare = read_acc_results('logs/pare/01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/01-09-2020_17-40-42_01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/val_accuracy_results.json')[:50]

plt.title('MPJPE Results / Epoch')
plt.xlabel('epochs')
plt.ylabel('PA-MPJPE')
plt.plot(acc_spin[:,1], label='spin')
plt.plot(acc_pare[:,1], label='pare')
plt.savefig('data/convergence_plot_pampjpe.png')
