from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points, transform_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


def nuscenes_sample_plot_lidartop(nusc,
                                  scene_idx: int,
                                  out_path: str,
                                  with_anns: bool = True,
                                  nsweeps: int = 1,
                                  underlay_map: bool = True,
                                  verbose: bool = False) -> None:
    my_scene = nusc.scene[scene_idx]
    last_sample_token = my_scene['last_sample_token']
    last_sample = nusc.get('sample', last_sample_token)
    sample_data_token = last_sample['data']['LIDAR_TOP']
    nusc.render_sample_data(sample_data_token, with_anns=with_anns, nsweeps=nsweeps, verbose=verbose, out_path=out_path,
                            underlay_map=underlay_map)


def plot_cam_effect_worse(data_path, boxes, camera_intrinsic):
    data = Image.open(data_path)
    # init axes
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    # show image
    ax.imshow(data)

    # show gt boxes.
    count = 0
    for box in boxes:
        count += 1
        if count % 2 != 0:
            continue
        c = np.array((255, 0, 0)) / 255.0  # red
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1.1)

    # count = 0
    # for box in boxes:
    #     count += 1
    #     if count % 2:
    #         continue
    #     c = np.array((0, 255, 0)) / 255.0
    #     t = np.random.uniform(-0.3, 0.3, 2)
    #     box.translate(np.array([t[0], t[1], 0]))
    #     r = np.random.uniform(-5, 5, 1)
    #     box.rotate(Quaternion([r[0], 0.0, 0.00, 0.0]))  # todo
    #     box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1.1)

    # Limit visible range.
    ax.set_xlim(0, data.size[0])
    ax.set_ylim(data.size[1], 0)

    ax.axis('off')
    # ax.set_title('{} {labels_type}'.format(
    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    plt.savefig('/home/leijiaming/painting/cam_effect_worse.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_cam_effect_better(data_path, boxes, camera_intrinsic):
    data = Image.open(data_path)
    # init axes
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    # show image
    ax.imshow(data)

    # show gt boxes.
    count = 0
    for box in boxes:
        count += 1
        if count % 2 != 0:
            continue
        c = np.array((255, 0, 0)) / 255.0  # red
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1.1)

    count = 0
    for box in boxes:
        count += 1
        if count % 2:
            continue
        c = np.array((0, 176, 80)) / 255.0
        t = np.random.uniform(-0.3, 0.3, 2)
        box.translate(np.array([t[0], t[1], 0]))
        r = np.random.uniform(-2, 2, 1)
        box.rotate(Quaternion([r[0], 0.0, 0.0, 0.0]))  # todo
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1.1)

    # Limit visible range.
    ax.set_xlim(0, data.size[0])
    ax.set_ylim(data.size[1], 0)

    ax.axis('off')
    # ax.set_title('{} {labels_type}'.format(
    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    plt.savefig('/home/leijiaming/painting/cam_effect_better.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_lidar_effect_worse(nusc, lidar_top_token, ):
    sd_record = nusc.get('sample_data', lidar_top_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=9)

    # Retrieve transformation matrices for reference point cloud.
    cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                  rotation=Quaternion(cs_record["rotation"]))

    # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    rotation_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
    viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Render map if requested.
    axes_limit = 40
    # nusc.explorer.render_ego_centric_map(sample_data_token=lidar_top_token, axes_limit=axes_limit, ax=ax)

    # Show point cloud.
    points = view_points(pc.points[:3, :], viewpoint, normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = (np.array((31, 36, 33)) / 255.0).reshape(1, -1)  # 象牙黑

    point_scale = 1.0
    scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')

    # Get boxes in lidar frame.
    _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=BoxVisibility.ANY,
                                       use_flat_vehicle_coordinates=True)

    # show gt boxes.
    for box in boxes:
        if 'car' in box.name:
            c = np.array(nusc.explorer.get_color(box.name)) / 255.0
            # c = np.array((255, 0, 0)) / 255.0
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1.5)

    # Show worser boxes.
    # for box in boxes:
    #     if 'car' in box.name:
    #         c = np.array((0, 255, 0)) / 255.0
    #         t = np.random.uniform(-1, 1, 2)
    #         box.translate(np.array([t[0], t[1], 0]))
    #         r = np.random.uniform(-10, 10, 1)
    #         box.rotate(Quaternion([r[0], 0.03, 0.13, 0.0]))  # todo
    #         box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1.5)

    # Limit visible range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    ax.axis('off')
    # ax.set_title('{} {labels_type}'.format(
    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    # plt.show()

    plt.savefig('/home/leijiaming/painting/lidar_effect_worser.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_lidar_effect_better(nusc, lidar_top_token, ):
    sd_record = nusc.get('sample_data', lidar_top_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=9)

    # Retrieve transformation matrices for reference point cloud.
    cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                  rotation=Quaternion(cs_record["rotation"]))

    # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    rotation_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
    viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Render map if requested.
    axes_limit = 40
    # nusc.explorer.render_ego_centric_map(sample_data_token=lidar_top_token, axes_limit=axes_limit, ax=ax)

    # Show point cloud.
    points = view_points(pc.points[:3, :], viewpoint, normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = (np.array((31, 36, 33)) / 255.0).reshape(1, -1)  # 象牙黑

    point_scale = 1.0
    scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')

    # Get boxes in lidar frame.
    _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=BoxVisibility.ANY,
                                       use_flat_vehicle_coordinates=True)

    # show gt boxes.
    for box in boxes:
        if 'car' in box.name:
            c = np.array((255, 0, 0)) / 255.0
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1.5)

    # Show better boxes.
    for box in boxes:
        if 'car' in box.name:
            c = np.array((0, 255, 0)) / 255.0
            t = np.random.uniform(-0.3, 0.3, 2)
            box.translate(np.array([t[0], t[1], 0]))
            r = np.random.uniform(-3, 3, 1)
            box.rotate(Quaternion([r[0], 0.0, 0.0, 0]))  # todo
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1.5)

    # Limit visible range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    ax.axis('off')
    # ax.set_title('{} {labels_type}'.format(
    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    # plt.show()

    plt.savefig('/home/leijiaming/painting/lidar_effect_better.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


if __name__ == '__main__':
    np.random.seed(3)
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
    nusc = NuScenes(version='v1.0-mini', dataroot='/dataset/nuscenes_mini', verbose=True)

    # scene index
    # scene_idx = 567
    scene_idx = 6
    scene_info = nusc.scene[scene_idx]
    # get sample info
    last_sample_info = nusc.get('sample', scene_info['last_sample_token'])
    # get sample_data_token cam_left
    cam_front_left_token = last_sample_info['data']['CAM_FRONT_LEFT']
    cam_front_left_sample_data_record = nusc.get('sample_data', cam_front_left_token)
    # get sample_data_toekn lidar_top
    lidar_top_token = last_sample_info['data']['LIDAR_TOP']
    # get sample_data cam_left
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_left_token, box_vis_level=BoxVisibility.ANY)

    # nusc.render_sample_data(cam_front_left_token, out_path='/home/leijiaming/painting/cam_left.png')
    # nusc.render_sample_data(lidar_top_token, out_path='/home/leijiaming/painting/lidar_top.png')

    plot_cam_effect_worse(data_path, boxes, camera_intrinsic)
    # plot_cam_effect_better(data_path, boxes, camera_intrinsic)
    # plot_lidar_effect_worse(nusc, lidar_top_token)
    # plot_lidar_effect_better(nusc, lidar_top_token)

    # sample index
    sample_idx = 430

    # nuscenes_sample_plot_lidartop(nusc, scene_idx, with_anns=False, nsweeps=3, underlay_map=False,
    #                               out_path=f'/home/leijiaming/painting/lidar-3sweep-{scene_idx}-notitle-nomap-noanns.png')
    # nuscenes_sample_plot_lidartop(nusc, scene_idx, with_anns=True, nsweeps=3, underlay_map=False,
    #                               out_path=f'/home/leijiaming/painting/lidar-3sweep-{scene_idx}-notitle-nomap-anns.png')
