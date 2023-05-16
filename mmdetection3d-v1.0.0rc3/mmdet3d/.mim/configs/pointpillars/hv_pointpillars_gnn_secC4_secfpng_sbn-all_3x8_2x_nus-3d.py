_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
# cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.25, 0.25, 8]
# model settings
model = dict(
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        _delete_=True,
        type='PillarFeatureNetGNN',
        in_channels=4,  # 
        max_num_points=64,  # 
        feat_channels=[64, 64],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        with_cluster_center=True,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False,),
    pts_backbone=dict(
        _delete_=True,
        type="SECONDC4",
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[16, 32, 64],
        conv_bias=False,),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPNG',
        in_channels=[16, 32, 64],
        upsample_strides=[1, 2, 4],
        out_channels=[32, 32, 32]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            ],
            sizes=[
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.4560939, 6.73778078, 2.73004906],  # truck
                [2.87427237, 12.01320693, 3.81509561],  # trailer
                [0.60058911, 1.68452161, 1.27192197],  # bicycle
                [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))

data=dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
)
