
Time_Embedd: [
    Linear(320, 1280),
    SiLU(),
    Linear(1280, 1280)
]

Label_Embedd: [
    Linear(2816, 1280)
    SiLU(),
    Linear(1280, 1280)
]

Input Blocks:
__BLOCK 1__
- TimestepEmbedSeq1: Conv2d(4, 320, 3x3, s11, p11)

- TimestepEmbed1: [
    ResBlock(320, 320, in_emb_proj=1280)
]
- TimestepEmbed2: [
    ResBlock(320, 320, in_emb_proj=1280)
]

- TimestepEmbedSeq2: Downsample(Conv2d(320, 320, 3x3, s22, p11))

__BLOCK 2__

- TimestepEmbed3: [
    ResBlock(320, 640, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(640, 640),
        BasicTransformerBlock(640, 640),
        BasicTransformerBlock(640, 640),
        Linear(640, 640)
    ]
]

- TimestepEmbed4: [
    ResBlock(640, 640, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(640, 640),
        BasicTransformerBlock(640, 640),
        BasicTransformerBlock(640, 640),
        Linear(640, 640)
    ]
]

- TimestepEmbedSeq3: Downsample(Conv2d(640, 640, 3x3, s22, p11))

__BLOCK 3__

- TimestepEmbed5: [
    ResBlock(640, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        Linear(1280, 1280)
    ]
]

- TimestepEmbed6: [
    ResBlock(1280, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        Linear(1280, 1280)
    ]
]

Middle Block:
- TimestepEmbed7: [
    ResBlock(1280, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        Linear(1280, 1280)
    ],
    ResBlock(1280, 1280, in_emb_proj=1280)
]

Output Blocks:
__BLOCK 1__
- TimestepEmbed7: [
    ResBlock(2560, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280)
        Linear(1280, 1280),
    ],
    ResBlock(1280, 1280, in_emb_proj=1280)
]

- TimestepEmbed8: [
    ResBlock(2560, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280)
        Linear(1280, 1280),
    ],
    ResBlock(1280, 1280, in_emb_proj=1280)
]

- TimestepEmbed9: [
    ResBlock(1920, 1280, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280),
        BasicTransformerBlock(1280, 1280)
        Linear(1280, 1280),
    ],
    ResBlock(1280, 1280, in_emb_proj=1280)
]

- TimestepEmbedSeq4: Upsample(Conv2d(1280, 1280, 3x3, s11, p11))

__BLOCK 2__
- TimestepEmbed10: [
    ResBlock(1920, 640, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(640, 640),
        BasicTransformerBlock(640, 640),
        BasicTransformerBlock(640, 640),
        Linear(640, 640)
    ]
]

- TimestepEmbed11: [
    ResBlock(1280, 640, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(640, 640),
        BasicTransformerBlock(640, 640),
        BasicTransformerBlock(640, 640),
        Linear(640, 640)
    ]
]

- TimestepEmbed12: [
    ResBlock(960, 640, in_emb_proj=1280),
    SpatialTransformer1: [
        Linear(640, 640),
        BasicTransformerBlock(640, 640),
        BasicTransformerBlock(640, 640),
        Linear(640, 640)
    ]
]

- TimestepEmbedSeq5: Upsample(Conv2d(640, 640, 3x3, s11, p11))

__BLOCK 3__

- TimestepEmbed13: [
    ResBlock(960, 320, in_emb_proj=1280)
]
- TimestepEmbed14: [
    ResBlock(640, 320, in_emb_proj=1280)
]
- TimestepEmbed15: [
    ResBlock(640, 320, in_emb_proj=1280)
]

- Out: [
    Conv2d(320, 4, 3x3, s11, p11)
]