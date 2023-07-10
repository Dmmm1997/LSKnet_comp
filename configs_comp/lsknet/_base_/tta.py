tta_model = dict(
    type='RotatedTTAModel',
    tta_cfg=dict(nms=dict(type='nms_rotated', iou_threshold=0.1), max_per_img=2000))

img_scales = [(1024, 1024)]
tta_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='mmdet.TestTimeAug',
        transforms=[
            [
                dict(type='mmdet.Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='mmdet.RandomFlip', prob=1.0),
                # dict(type='mmdet.RandomFlip', prob=1.0, direction=['horizontal']),
                # dict(type='mmdet.RandomFlip', prob=1.0, direction=['vertical']),
                # dict(type='mmdet.RandomFlip', prob=1.0, direction=['diagonal']),
                dict(type='mmdet.RandomFlip', prob=0.0),
            ],
            [
                dict(
                    type='mmdet.Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
            ],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]
