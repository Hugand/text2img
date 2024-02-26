import torch.nn as nn
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.model import Upsample

from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential

class DecodingUNetDownBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        time_embed_dim,
        num_head_channels,
        dropout=0.0,
        downsample=True,
        use_checkpoint=True,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.down_block = TimestepEmbedSequential(
            ResBlock(
                in_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            ResBlock(
                out_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            # SpatialTransformer(
            #     out_channels,
            #     out_channels // num_head_channels,
            #     num_head_channels,
            #     attn_type="softmax-xformers",
            #     use_checkpoint=use_checkpoint,
            #     use_linear=True
            # )
        )

    def forward(self, h, embeddings):
        return self.down_block(h, embeddings)

class DecodingUNetUpBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        time_embed_dim,
        num_head_channels,
        upsample=False,
        dropout=0.0,
        use_checkpoint=True,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.up_block = TimestepEmbedSequential(
            ResBlock(
                in_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                up=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            ResBlock(
                out_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                up=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            # SpatialTransformer(
            #     out_channels,
            #     out_channels // num_head_channels,
            #     num_head_channels,
            #     attn_type="softmax-xformers",
            #     use_checkpoint=use_checkpoint,
            #     use_linear=True
            # ),
            Upsample(
                in_channels=out_channels,
                with_conv=True
            ) if upsample else nn.Identity()
        )

    def forward(self, h, embeddings):
        return self.up_block(h, embeddings)
