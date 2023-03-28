import numpy as np
import joblib
from demo_pare_result import reorganize_pare
from collections import OrderedDict
from easydict import EasyDict as edict

def prepare_rendering_results(vibe_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            # import pdb;pdb.set_trace()
            # print(f'frame: {frame_id}')
            if person_data['joints3d'] is not None:
                frame_results[frame_id][person_id] = {
                    'pred_cam': person_data['pred_cam'][idx],
                    'orig_cam': person_data['orig_cam'][idx],
                    'verts': person_data['verts'][idx],
                    # 'joints2d': person_data['joints2d'][idx],
                    'betas': person_data['betas'][idx],
                    'pose': person_data['pose'][idx], 
                    'joints3d': person_data['joints3d'][idx],
                    'smpl_joints2d': person_data['smpl_joints2d'][idx],
                    'bboxes': person_data['bboxes'][idx]
                }
            else:
                print('false', person_data['orig_cam'][idx])
                frame_results[frame_id][person_id] = {
                    'pred_cam': np.zeros((3,)),
                    'orig_cam': np.zeros((3,)),
                    'verts': np.zeros((6890, 3)),
                    # 'joints2d': person_data['joints2d'][idx],
                    'betas': np.zeros((10,)),
                    'pose': np.zeros((24, 3, 3)), 
                    'joints3d': np.zeros((49, 3)),
                    'smpl_joints2d': np.zeros((49, 2)),
                    'bboxes': np.zeros((4,))
                }

    # naive depth ordering based on the scale of the weak perspective camera
    # find the largest one.
    # import pdb;pdb.set_trace()
    new_result = {}
    for frame_id, frame_data in enumerate(frame_results):
        # print(f'frame: {frame_id}')
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['orig_cam'][1] for k,v in frame_data.items()])
        # print(sort_idx)
        # frame_results[frame_id] = OrderedDict(
        #     {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        # )
        # * missing frame
        if len(sort_idx) == 0:
            # import pdb;pdb.set_trace()
            for key in new_result.keys():
                new_result[key].append(new_result[key][-1])
            continue
        # * find the largest one.
        tmp_frame_data = frame_data[list(frame_data.keys())[sort_idx[-1]]]
        for key, part_value in tmp_frame_data.items():
            if key not in new_result:
                new_result[key] = [part_value]
            else:
                new_result[key].append(part_value)
    # import pdb;pdb.set_trace()
    return edict(new_result)

    # return frame_results

if __name__ == '__main__':
    result_path = '/is/cluster/work/hyi/results/HDSR/PROX_qualitative_all/MPH8_00034_01/pare_results/Color_flip_/pare_output.pkl'
    pare_result = joblib.load(result_path)
    pare_result_dict = reorganize_pare(pare_result)
    tmp_pare_result_dict = prepare_rendering_results(pare_result, 2949)
    import pdb;pdb.set_trace()
