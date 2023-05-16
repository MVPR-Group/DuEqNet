import os
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def nuscenes_yaw_statics_plot():
    yaw = []
    # for sample in nusc.sample:
    #     for anno_token in sample['anns']:
    #         anno_record = nusc.get('sample_annotation', anno_token)
    #         yaw.append(Quaternion(anno_record['rotation']).yaw_pitch_roll[0])  # radian

    yaw = np.array(yaw)
    file_ = '/home/leijiaming/painting/rotation.npy'
    if os.path.exists(file_):
        yaw = np.load('/home/leijiaming/painting/rotation.npy')
    else:
        np.save('/home/leijiaming/painting/rotation.npy', yaw)
    # yaw = yaw / np.pi * 180  # degree

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(dpi=256)
    ax.hist(yaw, bins=10, edgecolor='black', color='steelblue', range=(-np.pi, np.pi),
            density=False)

    plt.tight_layout()
    plt.savefig('/home/leijiaming/painting/rotation.png')
    plt.close()


def nuscenes_annotation_statics(nusc):
    category = nusc.category
    for c in category:
        print(c['name'])
    """
    # 1 pedstrian: 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.wheelchair'
    'human.pedestrian.stroller', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer'
    'human.pedestrian.construction_worker'
    # 2 car: 'vehicle.car'
    # 3 motorcycle: 'vehicle.motorcycle'
    # 4 bicycle: 'vehicle.bicycle'
    # 5 bus: 'vehicle.bus.bendy', 'vehicle.bus.rigid'
    # 6 truck: 'vehicle.truck'
    # 7 construction_vehicle: 'vehicle.construction'
    # 8 trailer: 'vehicle.trailer'
    # 9 barrier: 'movable_object.barrier'
    # 10 traffic_cone: 'movable_object.trafficcone'
    """
    sample_annotations = nusc.sample_annotation

    annotation_statics = {
        'pedestrian': 0, 'car': 0, 'motorcycle': 0, 'bicycle': 0, 'bus': 0,
        'truck': 0, 'construction_vehicle': 0, 'trailer': 0, 'barrier': 0,
        'traffic_cone': 0
    }
    for anno in sample_annotations:
        if 'pedestrian' in anno['category_name']:
            annotation_statics['pedestrian'] += 1
        elif 'car' in anno['category_name']:
            annotation_statics['car'] += 1
        elif 'motorcycle' in anno['category_name']:
            annotation_statics['motorcycle'] += 1
        elif 'bicycle' in anno['category_name']:
            annotation_statics['bicycle'] += 1
        elif 'bus' in anno['category_name']:
            annotation_statics['bus'] += 1
        elif 'truck' in anno['category_name']:
            annotation_statics['truck'] += 1
        elif 'construction' in anno['category_name']:
            annotation_statics['construction_vehicle'] += 1
        elif 'trailer' in anno['category_name']:
            annotation_statics['trailer'] += 1
        elif 'barrier' in anno['category_name']:
            annotation_statics['barrier'] += 1
        elif 'trafficcone' in anno['category_name']:
            annotation_statics['traffic_cone'] += 1

    print(annotation_statics)
    # {'pedestrian': 222164, 'car': 493322, 'motorcycle': 12617, 'bicycle': 14572,
    # 'bus': 16321, 'truck': 88519, 'construction_vehicle': 14671, 'trailer': 24860,
    # 'barrier': 152087, 'traffic_cone': 97959}


if __name__ == '__main__':
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
    # nuscenes_annotation_statics(nusc)
    nuscenes_yaw_statics_plot()
    # yaw_statics = [50215, 180164, 89960, 94188, 157887, 44573, 172128, 108010, 103438, 165624]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    x = 1
    y = 2
