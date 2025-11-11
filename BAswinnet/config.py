auto_scale_lr = dict(base_batch_size=16, enable=False)
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmdet.models',
        'mmseg.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = ('data/datasets1', )
dataset_type = 'Datasets'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=5000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=False,
    custom_cfg=[
        dict(
            data_src='model_info',
            log_name='Parameters(M)',
            method_name='count_parameters',
            window_size=50),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        aspp_dilations=(
            1,
            6,
            12,
            18,
        ),
        aspp_out_channels=256,
        attn_drop_rate=0.15,
        convnext_depths=3,
        deep_stem=False,
        depths=(
            2,
            2,
            6,
            2,
        ),
        drop_path_rate=0.4,
        drop_rate=0.15,
        embed_dims=96,
        frozen_stages=-1,
        in_channels=3,
        init_cfg=dict(
            checkpoint='swin_tiny_patch4_window7_224_20220317-1cdeb081.pth',
            type='Pretrained'),
        layer_scale_init_value=1e-06,
        mask2former_in_channels=256,
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=(
            3,
            6,
            12,
            24,
        ),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        patch_size=4,
        pretrain_img_size=224,
        pretrained=None,
        qk_scale=None,
        qkv_bias=True,
        spatial_attn_kernel=9,
        strides=(
            4,
            2,
            2,
            2,
        ),
        style='pytorch',
        type='SwinEnd',
        use_abs_pos_embed=False,
        window_size=7,
        with_cp=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            256,
            256,
            256,
        ],
//
        num_classes=3,
        num_queries=100,
        num_transformer_feat_level=4,
        out_channels=256,
//
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=1.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=3.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=3.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.1,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
num_classes = 3
//
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.0001,
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=2000, start_factor=0.001,
        type='LinearLR'),
    dict(
        T_max=23000,
        begin=2000,
        by_epoch=False,
        end=25000,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='image/val', seg_map_path='annotation/val'),
        data_root='data/datasets1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='Datasets'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=1000)
//
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
//
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=2.0, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
                dict(direction='vertical', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
//
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'myconfigend/end'
