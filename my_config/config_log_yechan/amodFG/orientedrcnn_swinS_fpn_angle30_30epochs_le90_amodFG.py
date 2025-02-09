# 🚩 DEFAULT CONFIG ####################################################################################################
dataset_type = 'AMODFineGrainedDataset'
angles = [30]
data_root = 'data/AMOD_V1/'         # Important: should be ended with '/'
modality = 'EO'                     # 'eo' or 'ir'
img_extension = 'png'               # 'png' or 'jpg'
num_classes = 25                    # AMOD -> 13, AMOD_FG -> 25 (if civilian allowed? +1!)
load_from = None
resume_from = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90'
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
               pipeline=train_pipeline, version=angle_version, modality=modality, ext=img_extension,
               allow_civilian=allow_civilian),
    val=dict(type=dataset_type, data_root=data_root, ann_file='val.txt', img_prefix='train', angles=angles,
             pipeline=test_pipeline, version=angle_version, modality=modality, ext=img_extension,
             allow_civilian=allow_civilian),
    test=dict(type=dataset_type, data_root=data_root, ann_file='test.txt', img_prefix='test', angles=angles,
              pipeline=test_pipeline, version=angle_version, modality=modality, ext=img_extension,
              allow_civilian=allow_civilian)
)


# 🤖 MODEL CONFIG ######################################################################################################
model = dict(
    type='OrientedRCNN',
    backbone=dict(
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
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0
        )
    ),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)
            ),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0
            )
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0
        ),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000
        )
    )
)
