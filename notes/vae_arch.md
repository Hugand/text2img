<!-- Conv2d(in_ch, out_ch, kernel, stride, padding) -->
<!-- GroupNorm(num_groups, num_channels) -->
<!-- ResnetBlock(in_ch, out_ch) -->

```
Encoder:
- In: Conv2d(3, 128, 3x3, s11, p11)
- Down1: [
    ResnetBlock(128, 128),
    ResnetBlock(128, 128),
    <!-- NullAttention, -->
    Downsample(Conv2d(128, 128, 3x3, s22))
]
- Down2: [
    ResnetBlock(128, 256),
    ResnetBlock(256, 256),
    <!-- NullAttention, -->
    Downsample(Conv2d(256, 256, 3x3, s22))
]
- Down3: [
    ResnetBlock(256, 512),
    ResnetBlock(512, 512),
    <!-- NullAttention, -->
    Downsample(Conv2d(512, 512, 3x3, s22))
]
- Down4: [
    ResnetBlock(512, 512),
    ResnetBlock(512, 512),
    <!-- NullAttention, -->
]
- Mid: [
    ResnetBlock(512, 512),
    MemoryEfficientAttnBlock(512, 512)
    ResnetBlock(512, 512),
]
- Out: [
    GroupNorm(32, 512),
    Conv2d(512, 8, 3x3, s11, p11)
]

Decoder:
- In: Conv2d(4, 512, 3x3, s11, p11)
- Mid: [
    ResnetBlock(512, 512),
    MemoryEfficientAttnBlock(512, 512)
    ResnetBlock(512, 512),
]
- Up1: [
    ResnetBlock(512, 512),
    ResnetBlock(512, 512),
    ResnetBlock(512, 512),
    <!-- NullAttention, -->
    Upsample(Conv2d(512, 512, 3x3, s11, pp11))
]
- Up2: [
    ResnetBlock(512, 512),
    ResnetBlock(512, 512),
    ResnetBlock(512, 512),
    <!-- NullAttention, -->
    Upsample(Conv2d(512, 512, 3x3, s11, pp11))
]
- Up3: [
    ResnetBlock(512, 256),
    ResnetBlock(256, 256),
    ResnetBlock(256, 256),
    <!-- NullAttention, -->
    Upsample(Conv2d(256, 256, 3x3, s11, pp11))
]
- Up4: [
    ResnetBlock(256, 128),
    ResnetBlock(128, 128),
    ResnetBlock(128, 128),
]
- Out: [
    GroupNorm(32, 128),
    Conv2d(128, 3, 3x3, s11, p11)
]

Loss: Identity()
Regularization: DiagonalGaussianRegularizer()
Quant_conv: Conv2d(8, 8, 1x1, s11)
Post_quant_conv: Conv2d(4, 4, 1x1, s11)
```