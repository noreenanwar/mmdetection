_base_ = [
    #'../_base_/datasets/base_video_dataset.py',
    #'../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    #'./centernet_tta.py'
]
#dataset_type = 'BaseVideoDataset'
dataset_type = 'BaseVideoDataset'
# Define the classes for your dataset
#classes = ['Person', 'car', 'bus']
#metainfo=dict(classes=classes)
data_root = 'data/coco/'
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)
# model settings
model = dict(
    type='CenterNet',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, ),),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64),),
                #fuse_method='SUM'),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),),
                #fuse_method='SUM' ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),)),
                #fuse_method='SUM')),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')),
        #init_cfg=None ),
      

    neck=dict(
        type='FusionNeck',
  
        #in_channels=[64, 128, 256],,
        in_channels=[32,64, 128, 256],
        out_channels=64,  # Desired output channels after fusion
        num_outs=4  # Number of output feature maps, matching the input in this case
),


        #num_deconv_filters=(256, 128, 64),
        #num_deconv_kernels=(4, 4, 4),
        #use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=80,
        in_channels=64,
        feat_channels=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)

   
        ),
    #train_cfg=None,
    train_cfg=dict(
    rpn=dict(
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False))),

    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=50))
    
backend_args = None


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args,to_float32=True),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(
        type='RandomCenterCropPad',
        ratios=None,
        border=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','border',
                   'scale_factor'))
]
#classes = ['Person', 'car', 'bus']
#metainfo=dict(classes=classes)




train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_minitrain2017.json',
        data_prefix=dict(img='train2017/'),
        metainfo=dict(classes=('pedestrian', )),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        metainfo=dict(classes=('pedestrian', )),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=300, val_interval=1)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),  # warm up in the first 1000 iterations
    dict(type='MultiStepLR', by_epoch=True, milestones=[100, 200], gamma=0.1)  # reduce lr at 100 and 200 epochs
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # Adding TensorBoard logger
    ]
)


