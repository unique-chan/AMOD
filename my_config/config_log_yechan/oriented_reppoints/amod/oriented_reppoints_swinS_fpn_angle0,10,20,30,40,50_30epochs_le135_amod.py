# 🚩 DEFAULT CONFIG ####################################################################################################
dataset_type = 'AMODDataset'
angles = [0, 10, 20, 30, 40, 50]
data_root = 'data/AMOD_V1/'         # Important: should be ended with '/'
modality = 'EO'                     # 'eo' or 'ir'
img_extension = 'png'               # 'png' or 'jpg'
num_classes = 13                    # AMOD -> 13, AMOD_FG -> 25 (if civilian allowed? +1!)
load_from = None
resume_from = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le135'
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[16, 22]
)
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=-1) # save only when val mAP is best
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
                                      dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'


# 🧑‍🏫 TRAIN/VAL/TEST CONFIG #############################################################################################
# TIP: https://github.com/open-mmlab/mmdetection/issues/7680
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize',
        img_scale=[(1536, 1152), (2340, 1728)], # 0.8x - 1.2x  (1x: 1920x1440)
        multiscale_mode='range'),
    dict(type='RRandomCrop', crop_size=(1024, 1024), allow_negative_crop=False,
         crop_type='absolute', version=angle_version),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True), # Not allowed for val/test!
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1920, 1440)],
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # Not allowed for val/test!
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type=dataset_type, data_root=data_root, ann_file='train.txt', img_prefix='train', angles=angles,
               pipeline=train_pipeline, version=angle_version, modality=modality, ext=img_extension),
    val=dict(type=dataset_type, data_root=data_root, ann_file='val.txt', img_prefix='train', angles=angles,
             pipeline=test_pipeline, version=angle_version, modality=modality, ext=img_extension),
    test=dict(type=dataset_type, data_root=data_root, ann_file='test.txt', img_prefix='test', angles=angles,
              pipeline=test_pipeline, version=angle_version, modality=modality, ext=img_extension)
)


# 🤖 MODEL CONFIG ######################################################################################################
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RotatedRepPoints',
    backbone=dict(
        # _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     zero_init_residual=False,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_input',
    #     num_outs=5,
    #     norm_cfg=norm_cfg),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=num_classes,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        init_qua_weight=0.2,
        top_ratio=0.4),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxConvexIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.4),
        max_per_img=2000))