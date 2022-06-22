import joblib

def smpl_24joints():
    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')

    for k, v in smpl_segmentation['part2num'].items():
        print(k, v)

    parts = smpl_segmentation['smpl_index']

    parts[((parts >= 22) & (parts <= 24))] = 15
    parts[((parts >= 25) & (parts <= 39))] = 22
    parts[parts >= 40] = 23

    joblib.dump(
        {
            'smpl_index': parts,
            'part2num': {
                'Global': 0,
                'L_Thigh': 1,
                'R_Thigh': 2,
                'Spine': 3,
                'L_Calf': 4,
                'R_Calf': 5,
                'Spine1': 6,
                'L_Foot': 7,
                'R_Foot': 8,
                'Spine2': 9,
                'L_Toes': 10,
                'R_Toes': 11,
                'Neck': 12,
                'L_Shoulder': 13,
                'R_Shoulder': 14,
                'Head': 15,
                'L_UpperArm': 16,
                'R_UpperArm': 17,
                'L_ForeArm': 18,
                'R_ForeArm': 19,
                'L_Hand': 20,
                'R_Hand': 21,
                'L_Index1': 22,
                'R_Index1': 23
            }
        },
        'data/smpl_segmentation_24joints.pkl'
    )

def smpl_14joints():
    # calismiyor su anda
    # ama elleme kalsin
    # silme lazim olur
    smpl_segmentation = joblib.load('data/smpl_segmentation_24joints.pkl')

    for k, v in smpl_segmentation['part2num'].items():
        print(k, v)

    parts = smpl_segmentation['smpl_index']
    parts[parts == 0] = 2
    parts[parts == 1] = 1
    parts[parts == 2] = 2
    parts[parts == 3] = 1
    parts[parts == 4] = 1
    parts[parts == 5] = 1
    parts[parts == 6] = 1
    parts[parts == 7] = 1
    parts[parts == 8] = 1
    parts[parts == 9] = 1
    parts[parts == 10] = 1
    parts[parts == 11] = 1
    parts[parts == 12] = 1
    parts[parts == 13] = 1
    parts[parts == 14] = 1
    parts[parts == 15] = 1
    parts[parts == 16] = 1
    parts[parts == 17] = 1
    parts[parts == 18] = 1
    parts[parts == 19] = 1
    parts[parts == 20] = 1
    parts[parts == 21] = 1
    parts[parts == 22] = 1
    parts[parts == 23] = 1

    joblib.dump(
        {
            'smpl_index': parts,
            'part2num': {
                'Global': 0,
                'L_Thigh': 1,
                'R_Thigh': 2,
                'Spine': 3,
                'L_Calf': 4,
                'R_Calf': 5,
                'Spine1': 6,
                'L_Foot': 7,
                'R_Foot': 8,
                'Spine2': 9,
                'L_Toes': 10,
                'R_Toes': 11,
                'Neck': 12,
                'L_Shoulder': 13,
                'R_Shoulder': 14,
                'Head': 15,
                'L_UpperArm': 16,
                'R_UpperArm': 17,
                'L_ForeArm': 18,
                'R_ForeArm': 19,
                'L_Hand': 20,
                'R_Hand': 21,
                'L_Index1': 22,
                'R_Index1': 23
            }
        },
        'data/smpl_segmentation_24joints.pkl'
    )

if __name__ == '__main__':
    smpl_24joints()
    smpl_14joints()