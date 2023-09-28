model = dict(
    type = 'EncoderDecoder',
    pretrained = None,
    backbone = dict(
        type='DINOv2',
        model='b14',
    ),
    decode_head = dict(
        type='FCNHead',
        in_channels=768,
        channels=768,
        num_convs=0,
        num_classes=20,
        dropout_ratio=0,
        concat_input=False,
        input_transform='full_batch',
    ),
)