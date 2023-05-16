_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.25, 0.25, 8]
model = dict(
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
    pts_bbox_head=dict(
        _delete_=True,
        type='FreeAnchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25])))

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
)
