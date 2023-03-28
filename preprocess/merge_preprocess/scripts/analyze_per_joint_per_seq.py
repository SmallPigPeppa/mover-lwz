import sys
import numpy as np
import joblib
from pare.utils.kp_utils import get_common_joint_names
from pare.core.constants import pw3d_occluded_sequences, pw3d_test_sequences

joint_names = get_common_joint_names()
occluded_sequences = pw3d_occluded_sequences
test_sequences = pw3d_test_sequences

report_joints = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'head']
joint_names[-2:] = ['head', 'head']

def main(file):
    results = joblib.load(file)

    ######## TEST SET #########

    seq_names = [x.split('/')[-2] for x in results['imgname']]

    mpjpe = []
    pampjpe = []

    ss = []
    for idx, seq in enumerate(seq_names):

        if seq in test_sequences:
            mpjpe.append(results['mpjpe'][idx].mean())
            pampjpe.append(results['pampjpe'][idx].mean())
            ss.append(seq)

    print(len(list(set(ss))))
    mpjpe = np.array(mpjpe)
    pampjpe = np.array(pampjpe)

    print(results['mpjpe'].shape, mpjpe.shape)
    print(results['pampjpe'].shape, pampjpe.shape)

    print(f'MPJPE on Test Set: {np.array(mpjpe).mean() * 1000}')
    print(f'PA-MPJPE on Test Set: {np.array(pampjpe).mean() * 1000}')

    ######## OCCLUSION #########
    # occluded error
    seq_names = [x.split('/')[-2] for x in results['imgname']]

    mpjpe = []
    pampjpe = []

    ss = []
    for idx, seq in enumerate(seq_names):
        seq = '_'.join(seq.split('_')[:2])

        if seq in occluded_sequences:
            mpjpe.append(results['mpjpe'][idx].mean())
            pampjpe.append(results['pampjpe'][idx].mean())
            ss.append(seq)

    print(len(list(set(ss))))
    mpjpe = np.array(mpjpe)
    pampjpe = np.array(pampjpe)

    print(results['mpjpe'].shape, mpjpe.shape)
    print(results['pampjpe'].shape, pampjpe.shape)

    print(f'MPJPE on Occluded Sequences: {np.array(mpjpe).mean() * 1000}')
    print(f'PA-MPJPE on Occluded Sequences: {np.array(pampjpe).mean() * 1000}')

    ######## PER JOINT #########
    # per joint error
    print('***** MPJPE *****')
    mpjpe = results['mpjpe'].mean(0) * 1000
    # for j, e in zip(joint_names, mpjpe):
    #     print(f'{j}\t: {e:.2f}')
    #
    # print('***** PA-MPJPE *****')
    pampjpe = results['pampjpe'].mean(0) * 1000
    # for j, e in zip(joint_names, pampjpe):
    #     print(f'{j}\t: {e:.2f}')

    for j in report_joints:
        print(j.capitalize(), end=',')
    print()

    for j in report_joints:
        e = mpjpe[[idx for idx, x in enumerate(joint_names) if x.endswith(j)]].mean()
        print(f'{e:.2f}', end=',')
    print()

    for j in report_joints:
        e = pampjpe[[idx for idx, x in enumerate(joint_names) if x.endswith(j)]].mean()
        print(f'{e:.2f}', end=',')
    print()

    ######## PER ACTION #########

    mpjpe = []
    pampjpe = []

    for idx, seq in enumerate(seq_names):
        mpjpe.append(results['mpjpe'][idx].mean())
        pampjpe.append(results['pampjpe'][idx].mean())

    mpjpe = np.array(mpjpe)
    pampjpe = np.array(pampjpe)


    action_labels = np.load('data/clustering/rawpose_n20_3dpw_labels.npy')

    actions = {
        'Sit': [2,6,19],
        'Crouch': [7,4,12,14],
        'Stand': [1,3,5,9,10,11,13,15,18],
        'Other': [0,8,16,17]
    }

    print('Sit,Crouch,Stand,Other')

    for action, act_labels in actions.items():
        act_err = []

        for al in act_labels:
            act_err.append(mpjpe[action_labels == al])

        act_err = np.concatenate(act_err)
        print(f'{act_err.mean()*1000:.2f}', end=',')
    print()

    for action, act_labels in actions.items():
        act_err = []

        for al in act_labels:
            act_err.append(pampjpe[action_labels == al])

        act_err = np.concatenate(act_err)
        print(f'{act_err.mean()*1000:.2f}', end=',')
    print()



if __name__ == '__main__':
    file = 'logs/pare/30.08-pare_eft_wo_iter_pretrained_cliploss_evaluation/31-08-2020_12-48-37_30.08-pare_eft_wo_iter_pretrained_cliploss_evaluation_dataset.valds-3dpw/evaluation_results_3dpw.pkl'

    main(sys.argv[1])