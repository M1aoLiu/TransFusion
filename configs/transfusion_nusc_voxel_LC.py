point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] # 点云数据取值范围
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
] # 类别名称 
voxel_size = [0.075, 0.075, 0.2] # voxel大小
out_size_factor = 8 # 下采样大小，输出图像为输入的1/8 对应后面的out_indices和num_stages(原图，以及3次下采样的图)
evaluation = dict(interval=1) # 验证频率
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
# 输入数据类型：只使用lidar和camera，不使用毫米波雷达、地图和外参矩阵
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (800, 448) # 输入图像大小(下面的MyResize会将输入图像Resize)
num_views = 6 # 使用相机视角的个数
# 图像归一化参数
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # 使用lidar数据的前5个维度(xyz+反射强度+激光雷达扫描环编号)
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    # 使用非关键帧sweeps
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10, # 这表示我们要加载的扫描数量。在这里，我们加载了 10 个时间步的点云数据。
        use_dim=[0, 1, 2, 3, 4],
    ),
    # 加载Annotations
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # 加载多视角下拍摄的图像
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # 根据point_cloud_range限定点云数据的取值范围
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # 根据point_cloud_range限定物体的取值范围
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 根据class_names限定物体类别
    dict(type='ObjectNameFilter', classes=class_names),
    # 点云点顺序随机打乱
    dict(type='PointShuffle'),
    # 根据img_scale
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    # 图像归一化
    dict(type='MyNormalize', **img_norm_cfg),
    # 填充图像padding，确保能被32整除
    dict(type='MyPad', size_divisor=32),
    # 不同数据项的尺寸可能不一致，例如点云中点的数量、真实标注框的尺寸.
    # DefaultFormatBundle3D 的目标是将这些不同尺寸的数据项统一格式化，以便后续处理.
    # 为了处理不同尺寸的数据，mmdetection3d 引入了 DataContainer 类型。
    # DataContainer 可以帮助收集和分发不同尺寸的数据
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # 在目标检测任务中，我们需要将不同数据项（例如点云、真实的 3D 边界框、类别标签等）整合到一起，以便输入到模型中。
    # Collect3D 的目标是从不同数据源中收集这些数据项('points', 'img', 'gt_bboxes_3d', 'gt_labels_3d')，并将它们组织成一个统一的数据结构
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    # 多尺寸的数据增强
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False, # 不翻转
        transforms=[
            # 世界坐标系下的旋转缩放和平移
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            # 3D随即翻转
            dict(type='RandomFlip3D'),
            # 图像的Resize、归一化和padding
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            num_views=num_views,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='TransFusionDetector',
    freeze_img=True,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    # 图像特征提取
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4, #  ResNet50,特征提取后输出结果有4层不同大小的features 
        out_indices=(0, 1, 2, 3), # 此处为4层输出特征图的索引
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    # 图像特征融合和增强
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5), # num_outs表示FPN图像特征融合模块的输出数量，用于更好地捕捉不同尺度的目标
    pts_voxel_layer=dict(
        max_num_points=10, # 一个voxel最多10个点
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440], # 描述了一个具有41层深度和1440x1440分辨率的稀疏点云,依次表示z,y,x方向的体素数
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True, # 表示是否融合图像信息。如果为 True，则将图像特征与 LiDAR 特征进行融合
        num_views=num_views,
        in_channels_img=256, # 图像特征的输入通道数
        out_size_factor_img=4,
        num_proposals=200, # 预测的边界框数量
        auxiliary=True,
        in_channels=256 * 2, # 输入通道数，考虑了图像和 LiDAR 特征
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1, # 解码器的层数
        num_heads=8, # 注意力头的数量
        learnable_query_pos=False, # 是否学习查询位置
        initialize_by_heatmap=True, # 是否通过热图初始化
        nms_kernel_size=3, # 非极大值抑制的卷积核大小
        ffn_channel=256, # 前馈网络通道数
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        # 在三维目标检测中，我们通常需要预测多个目标属性，例如目标的中心点、高度、宽度、深度、旋转角度、速度等。
        # 这些属性通常被称为“头部”（heads），每个头部负责预测一个特定的目标属性。
        # 包含不同任务的头部信息
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)), 
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 6
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
# load_from = 'checkpoints/fusion_voxel0075_R50.pth'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)] # 循环迭代进行。此处为循环训练1个epoch，如果为[('train', 1),('val', 1)]则为训练1epoch验证1epoch循环进行
gpu_ids = range(0, 1)
freeze_lidar_components = True
find_unused_parameters = True
