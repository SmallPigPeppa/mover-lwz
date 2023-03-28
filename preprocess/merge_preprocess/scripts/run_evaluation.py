import os
import sys
import subprocess

if __name__ == '__main__':
    eval_datasets = ['3dpw-all', 'mpi-inf-3dhp', 'h36m-p2']

    eval_cmds = [
        {
            'name': 'spin-eft',
            'dir': 'logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/',
            'ckpt': 'tb_logs_pare/0_af6934cb2bdf49bc892a61c0306526a9/checkpoints/epoch=78.ckpt',
        },
        # {
        #     'name': 'pare-eft',
        #     'dir': 'logs/pare/01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/01-09-2020_17-40-42_01.09-pare_eft_wo_iter_pretrained_cliploss_mpii3d_fix/',
        #     'ckpt': 'tb_logs_pare/0_c2083df7f5964bfa881e0b5097c76639/checkpoints/epoch=92.ckpt',
        # },
        # {
        #     'name': 'pare-eft-synthoccfinetune',
        #     'dir': 'logs/pare/03.09-pare_synth_occ_finetune/03-09-2020_15-46-30_03.09-pare_synth_occ_finetune/',
        #     'ckpt': 'tb_logs_pare/0_4d1d75ce0c28412d980f9eaadc624f09/checkpoints/epoch=49.ckpt',
        # },
        # {
        #     'name': 'pare-eft-synthoccpretrained',
        #     'dir': 'logs/pare/03.09-pare_synth_occ_pretrained/03-09-2020_15-41-09_03.09-pare_synth_occ_pretrained/',
        #     'ckpt': 'tb_logs_pare/0_68b05bf24bd24c1898d727a297cd6585/checkpoints/epoch=22.ckpt',
        # },
        # {
        #     'name': 'pare-eft-usekpfeats',
        #     'dir': 'logs/pare/03.09-pare_use_kp_feats_for_smpl_regression/03-09-2020_12-47-25_03.09-pare_use_kp_feats_for_smpl_regression/',
        #     'ckpt': 'tb_logs_pare/0_616ac2ca9b9946c89804aabb7cbf7802/checkpoints/epoch=86.ckpt',
        # }
    ]

    total = len(eval_cmds) * len(eval_datasets)
    counter = 1
    for eval_cmd in eval_cmds:
        for ds in eval_datasets:
            cmd = [
                'python', 'eval.py',
                '--cfg', os.path.join(eval_cmd['dir'], 'config_to_run.yaml'),
                '--ckpt', os.path.join(eval_cmd['dir'], eval_cmd['ckpt']),
                '--val_ds', ds,
            ]
            print(f'{counter}/{total} -- Executing', ' '.join(cmd))

            subprocess.call(cmd)
            counter += 1
