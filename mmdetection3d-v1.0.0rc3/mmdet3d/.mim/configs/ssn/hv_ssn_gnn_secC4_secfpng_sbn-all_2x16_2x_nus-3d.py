_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
# Note that the order of class names should be consistent with
# the following anchors' order
point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.25, 0.25, 8]
class_names = [
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'car',
    'truck', 'trailer', 'bus', 'construction_vehicle'
]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, classes=class_names),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# model settings
voxel_size = [0.25, 0.25, 8]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNetGNN',
        in_channels=4,  # 
        max_num_points=20,  # 
        feat_channels=[64, 64],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        with_cluster_center=True,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False,),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='SECONDC4',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[16, 32, 64],
        conv_bias=False,
    ),
    pts_neck=dict(
        type='SECONDFPNG',
        in_channels=[16, 32, 64],
        upsample_strides=[1, 2, 4],
        out_channels=[32, 32, 32]
    ),
    pts_bbox_head=dict(
        type='ShapeAwareHead',
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGeneratorPerCls',
            ranges=[[-50, -50, -1.67339111, 50, 50, -1.67339111],
                    [-50, -50, -1.71396371, 50, 50, -1.71396371],
                    [-50, -50, -1.61785072, 50, 50, -1.61785072],
                    [-50, -50, -1.80984986, 50, 50, -1.80984986],
                    [-50, -50, -1.76396500, 50, 50, -1.76396500],
                    [-50, -50, -1.80032795, 50, 50, -1.80032795],
                    [-50, -50, -1.74440365, 50, 50, -1.74440365],
                    [-50, -50, -1.68526504, 50, 50, -1.68526504],
                    [-50, -50, -1.80673031, 50, 50, -1.80673031],
                    [-50, -50, -1.64824291, 50, 50, -1.64824291]],
            sizes=[
                [0.60058911, 1.68452161, 1.27192197],  # bicycle
                [0.76279481, 2.09973778, 1.44403034],  # motorcycle
                [0.66344886, 0.72564370, 1.75748069],  # pedestrian
                [0.39694519, 0.40359262, 1.06232151],  # traffic cone
                [2.49008838, 0.48578221, 0.98297065],  # barrier
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.45609390, 6.73778078, 2.73004906],  # truck
                [2.87427237, 12.01320693, 3.81509561],  # trailer
                [2.94046906, 11.1885991, 3.47030982],  # bus
                [2.73050468, 6.38352896, 3.13312415]  # construction vehicle
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=False),
        tasks=[
            dict(
                num_class=2,
                class_names=['bicycle', 'motorcycle'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=1,
                class_names=['pedestrian'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=2,
                class_names=['traffic_cone', 'barrier'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=1,
                class_names=['car'],
                shared_conv_channels=(64, 64, 64),
                shared_conv_strides=(2, 1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=4,
                class_names=[
                    'truck', 'trailer', 'bus', 'construction_vehicle'
                ],
                shared_conv_channels=(64, 64, 64),
                shared_conv_strides=(2, 1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01))
        ],
        assign_per_class=True,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # bicycle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # motorcycle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                dict(  # pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # traffic cone
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # barrier
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # truck
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # trailer
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # bus
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # construction vehicle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500))
)
