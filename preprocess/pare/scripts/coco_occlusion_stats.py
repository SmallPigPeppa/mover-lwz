import os
import sys
import cv2
import pylab
import joblib
import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from pare.utils.kp_utils import get_coco_joint_names
from pare.core.config import COCO_ROOT

def collect_stats(split='val2014', debug=False):
    # data_dir = '/is/cluster/work/mkocabas/projects/pare/data/dataset_folders/coco'

    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    instance_ann_file=f'{COCO_ROOT}/annotations/instances_{split}.json'

    # initialize COCO api for instance annotations
    coco_instance = COCO(instance_ann_file)

    kps_ann_file = f'{COCO_ROOT}/annotations/person_keypoints_{split}.json'
    coco_kps = COCO(kps_ann_file)

    # get all images containing given categories, select one at random
    catIds = coco_instance.getCatIds(catNms=['person'])
    imgIds = coco_instance.getImgIds(catIds=catIds)
    img_data = coco_instance.loadImgs(imgIds)

    cats = coco_instance.loadCats(coco_instance.getCatIds())
    categories = {cat['id']: cat['name'] for cat in cats}

    joint_names = get_coco_joint_names()

    occluder_dict = dict.fromkeys(joint_names)

    for k in occluder_dict.keys():
        occluder_dict[k] = []

    structs = ['object_with_mask', 'person_bbox', 'obj_bbox', 'occluded_joint', 'obj_class']
    occluders = dict.fromkeys(structs)
    for k in occluders.keys():
        occluders[k] = []

    for i in tqdm(range(len(img_data))):

        occluder_exist = []

        img = img_data[i]
        imgname = os.path.join(COCO_ROOT, split, img['file_name'])

        image = io.imread(imgname)

        instance_ann_ids = coco_instance.getAnnIds(imgIds=img['id'], iscrowd=None)
        instance_anns = coco_instance.loadAnns(instance_ann_ids)
        # coco_instance.showAnns(instance_anns)

        kp_ann_ids = coco_kps.getAnnIds(imgIds=img['id'], iscrowd=None)
        kp_anns = coco_kps.loadAnns(kp_ann_ids)

        anns_img = np.zeros((img['height'], img['width']))
        for ann in instance_anns:
            if ann['category_id'] == 1:
                continue

            anns_img = np.maximum(anns_img, coco_instance.annToMask(ann) * ann['category_id'])

            ann_mask = coco_instance.annToMask(ann) * 255
            obj_bbox = np.array(ann['bbox']).astype(int)

            counter = 0
            for kp_ann_idx, kp_ann in enumerate(kp_anns):
                keypoints = np.array(kp_ann['keypoints']).reshape(-1, 3)
                person_bbox = np.array(kp_ann['bbox']).astype(int)

                for kp_idx, k in enumerate(keypoints):
                    if k[2] == 1:
                        if ann_mask[k[1], k[0]] > 0:
                            occluder_dict[joint_names[kp_idx]].append(
                                (categories[ann['category_id']], obj_bbox[3], person_bbox[3]/obj_bbox[3])
                            )
                            occluder_exist.append([joint_names[kp_idx], categories[ann['category_id']], k])

                            if counter > 0:
                                continue

                            if obj_bbox[2] * obj_bbox[3] < 5000:
                                continue

                            if len(image.shape) == 2:
                                # skip gray scale images
                                print('Gray scale image, skipping...')
                                continue

                            object_image = image[obj_bbox[1]:obj_bbox[1]+obj_bbox[3],
                                                 obj_bbox[0]:obj_bbox[0]+obj_bbox[2]]
                            object_mask = ann_mask[obj_bbox[1]:obj_bbox[1]+obj_bbox[3],
                                                   obj_bbox[0]:obj_bbox[0]+obj_bbox[2]]

                            # Reduce the opacity of the mask along the border for smoother blending
                            eroded = cv2.erode(object_mask, structuring_element)
                            object_mask[eroded < object_mask] = 192

                            try:
                                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
                            except Exception as e:
                                print(e)
                                import IPython; IPython.embed()
                                continue

                            if debug:
                                plt.title(f'{categories[ann["category_id"]]}, {ann["category_id"]}, '
                                          f'area: {obj_bbox[2] * obj_bbox[3]}')
                                plt.imshow(object_with_mask); plt.show()

                            occluders['object_with_mask'].append(object_with_mask)
                            occluders['person_bbox'].append(person_bbox)
                            occluders['obj_bbox'].append(obj_bbox)
                            occluders['occluded_joint'].append(joint_names[kp_idx])
                            occluders['obj_class'].append(categories[ann['category_id']])

                            counter += 1

        # if i == 200:
        #     break

        # if debug and len(occluder_exist) > 0:
        #     im = io.imread(imgname)
        #     plt.imshow(im)
        #     coco_kps.showAnns(kp_anns)
        #     coco_instance.showAnns(instance_anns)
        #     print(occluder_exist)
        #     plt.show()
        #
        #     plt.imshow(object_with_mask)
        #     plt.show()

    occluders['person_bbox'] = np.array(occluders['person_bbox'])
    occluders['obj_bbox'] = np.array(occluders['obj_bbox'])
    occluders['occluded_joint'] = np.array(occluders['occluded_joint'])
    occluders['obj_class'] = np.array(occluders['obj_class'])

    # print(f'Saving coco-{split} occlusion statistics')
    # joblib.dump(occluder_dict, f'data/occlusion_augmentation/coco_{split}_occlusion_stats.pkl')
    print(f'Saving occluder objects!')
    print(f'{len(occluders["obj_class"])} objects found!')
    occluders['stats'] = copy_data_to_spin_joints(occluder_dict)
    joblib.dump(occluders, f'data/occlusion_augmentation/coco_{split}_occluders.pkl')
    # np.savez(
    #     f'data/occlusion_augmentation/coco_{split}_occluders.npz',
    #     object_with_mask=occluders['object_with_mask'],
    #     person_bbox=occluders['person_bbox'],
    #     obj_bbox=occluders['obj_bbox'],
    #     occluded_joint=occluders['occluded_joint'],
    #     obj_class=occluders['obj_class'],
    # )

    # import IPython; IPython.embed(); exit()


def copy_data_to_spin_joints(coco_stats):
    spin_stats = {
        'rankle': coco_stats['rankle'],
        'rknee': coco_stats['rknee'],
        'rhip': coco_stats['rhip'],
        'lhip': coco_stats['lhip'],
        'lknee': coco_stats['lknee'],
        'lankle': coco_stats['lankle'],
        'rwrist': coco_stats['rwrist'],
        'relbow': coco_stats['relbow'],
        'rshoulder': coco_stats['rshoulder'],
        'lshoulder': coco_stats['lshoulder'],
        'lelbow': coco_stats['lelbow'],
        'lwrist': coco_stats['lwrist'],
        'neck': coco_stats['nose'],
        'headtop': coco_stats['nose'],
        'hip': coco_stats['rhip'],
        'thorax': coco_stats['lshoulder'],
        'Spine (H36M)': coco_stats['lhip'],
        'Jaw (H36M)': coco_stats['nose'],
        'Head (H36M)': coco_stats['nose'],
        'nose': coco_stats['nose'],
        'leye': coco_stats['leye'],
        'reye': coco_stats['reye'],
        'lear': coco_stats['lear'],
        'rear': coco_stats['rear'],
    }
    return spin_stats



if __name__ == '__main__':

    if len(sys.argv) > 1:
        collect_stats(sys.argv[1], debug=int(sys.argv[2]))
    else:
        collect_stats('train2014', debug=False)