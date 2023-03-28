import os
import cv2
import time
import joblib
import skvideo.io
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from pare.utils.vis_utils import  draw_skeleton

MIN_NUM_FRAMES = 300
MIN_NUM_KP = 15
MIN_THRESHOLD = 0.7
MIN_NUM_PERSON = 6

vid_folder = '/ps/scratch/mkocabas/data/Kinetics-400/videos'
folder = '/ps/scratch/mkocabas/data/Kinetics-400/posetrack_annotations'


def filter(debug=False):

    actions = sorted(os.listdir(folder))

    valid_annotation_files = []
    total_num_frames = 0

    for action in actions:
        videos = sorted(os.listdir(osp.join(folder, action)))

        for vid in videos:
            data = joblib.load(osp.join(folder, action, vid))
            if len(data.keys()) < MIN_NUM_PERSON: # discard crowded videos
                for person_id in data.keys():
                    num_frames = data[person_id]['frames'].shape[0]
                    if num_frames >= MIN_NUM_FRAMES:
                        joints2d = data[person_id]['joints2d']
                        vis = joints2d[:,:,2] > MIN_THRESHOLD
                        vis = vis.sum() / num_frames

                        if vis > MIN_NUM_KP:
                            total_num_frames += num_frames
                            valid_annotation_files.append(f'{action}/{vid}/{person_id}')

                            if debug:
                                # debug
                                video_file = osp.join(vid_folder, action, vid.replace('.pkl', '.mp4'))
                                vf = skvideo.io.vread(video_file)
                                for i in data[person_id]['frames']:
                                    frame = vf[i]
                                    kp_2d = data[person_id]['joints2d'][i]
                                    kp_2d[:,2][kp_2d[:,2] > MIN_THRESHOLD] = 1.
                                    frame = draw_skeleton(frame, kp_2d,
                                                          dataset='staf', unnormalize=False, thickness=6)
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                                    cv2.imshow(action + '/' + vid, frame)

                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        break
                                cv2.destroyAllWindows()

        print(action, total_num_frames)

    print('Number of frames: ', total_num_frames)
    print('Saving valid annotation files')
    joblib.dump(valid_annotation_files, 'kinetics_valid_annotations.pkl')

def visualize_filtered_videos():
    ann_files = joblib.load('/ps/scratch/mkocabas/data/Kinetics-400/kinetics_valid_annotations_1.8M.pkl')
    np.random.shuffle(ann_files)

    print('Number of videos', len(ann_files))

    for ann in ann_files:
        print(ann)
        action, fn, person_id = ann.split('/')
        person_id = int(person_id)
        data = joblib.load(osp.join(folder, action, fn))

        video_file = osp.join(vid_folder, action, fn.replace('.pkl', '.mp4'))
        vf = skvideo.io.vread(video_file).astype(float)

        for i in data[person_id]['frames']:
            frame = vf[i] / 255.
            kp_2d = data[person_id]['joints2d'][i]
            kp_2d[:, 2][kp_2d[:, 2] > MIN_THRESHOLD] = 1.
            frame = draw_skeleton(frame, kp_2d, dataset='staf', unnormalize=False, thickness=4) * 255
            frame = frame.astype(np.uint8)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow(action + '/' + fn, frame)
            # time.sleep(0.033)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def get_bbox_from_kp2d(kp_2d):
    # get bbox
    if len(kp_2d.shape) > 2:
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)

    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox


def analyze_videos():
    ann_files = joblib.load('/ps/scratch/mkocabas/data/Kinetics-400/kinetics_valid_annotations_1.8M.pkl')
    np.random.shuffle(ann_files)

    print('Number of videos', len(ann_files))

    empty_counter = 0

    heights = []
    for ann in tqdm(ann_files):
        # print(ann)
        action, fn, person_id = ann.split('/')
        person_id = int(person_id)
        data = joblib.load(osp.join(folder, action, fn))

        # video_file = osp.join(vid_folder, action, fn.replace('.pkl', '.mp4'))
        # vf = skvideo.io.vread(video_file).astype(float)

        d = data[person_id]['joints2d']
        h = []
        for i in data[person_id]['frames']:
            d_ = d[i]
            d_ = d_[d_[:, 2] > MIN_THRESHOLD, :]

            if d_.shape[0] < 1:
                empty_counter += 1
                print(f'{empty_counter} empty arrays detected...')
                continue

            bb = get_bbox_from_kp2d(d_)
            h.append(bb[-1])

        heights += h

    plt.figure(figsize=(10, 10))
    plt.title('Kinetics-400 bbox size histogram')
    plt.xlabel('bbox size in pixels')
    plt.ylabel('frequency')
    plt.hist(heights, bins='auto')
    plt.show()
    import IPython; IPython.embed(); exit()


if __name__ == '__main__':
    # filter(debug=False)
    # visualize_filtered_videos()
    analyze_videos()