import os
import json
import os.path as osp
import numpy as np
import argparse
import pprint
import shutil
from pare.core.config import update_hparams
from flatten_dict import flatten
from torch.utils.tensorboard import SummaryWriter
import subprocess

def summary_writer(result_folder, new_log_dir):
    child = result_folder.split('/')[-1]
    print(child)
    log_f = osp.join(new_log_dir, child)

    writer = SummaryWriter(log_dir=log_f) # comment=child

    acc_results = json.load(open(os.path.join(result_folder, 'val_accuracy_results.json')))
    hparams = flatten(update_hparams(os.path.join(result_folder, 'config_to_run.yaml')), reducer='path')

    hparams_cp = hparams.copy()
    for k, v in hparams.items():
        if v is None:
            hparams_cp[k] = ''

    hparams = hparams_cp

    acc_arr = []
    for acc in acc_results:
        if acc[1]['val_mpjpe'] < 10:
            continue

        writer.add_scalar('accuracy/mpjpe', acc[1]['val_mpjpe'], acc[0])
        writer.add_scalar('accuracy/pa-mpjpe', acc[1]['val_pampjpe'], acc[0])
        acc_arr.append([acc[1]['val_mpjpe'], acc[1]['val_pampjpe']])

    accuracy = np.array(acc_arr)

    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            'hparams/mpjpe': accuracy[:, 0].min(),
            'hparams/pa-mpjpe': accuracy[:, 1].min()
        }
    )

def main(args):
    result_dir = args.dirs
    log_dir = args.log_dir
    if osp.isdir(log_dir):
        shutil.rmtree(log_dir)

    for root, dirs, files in os.walk(result_dir, topdown=False):
        for name in files:
            if name.endswith('val_accuracy_results.json'):
                summary_writer(result_folder=root, new_log_dir=log_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dirs', type=str, help='list of directories separated by comma')
    parser.add_argument('--log_dir', type=str, help='log_dir', default='logs/.temp_tensorboard')
    args = parser.parse_args()
    main(args)

    subprocess.call([
        'tensorboard', '--logdir', args.log_dir
    ])