import torch.nn as nn

from generative_models.sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential

class DecodingUNetDownBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        time_embed_dim,
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
                out_channels=in_channels,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            ResBlock(
                in_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                down=downsample,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

    def forward(self, h, embeddings):
        return self.down_block(h, embeddings)

class DecodingUNetUpBlock(nn.Module):
    def __init__(self,
        in_channels,
        inner_channels,
        out_channels,
        time_embed_dim,
        upsample=False,
        dropout=0.0,
        use_checkpoint=True,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.down_block = TimestepEmbedSequential(
            ResBlock(
                in_channels,
                time_embed_dim,
                dropout,
                out_channels=inner_channels,
                up=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            ResBlock(
                inner_channels,
                time_embed_dim,
                dropout,
                out_channels=inner_channels,
                up=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            ResBlock(
                inner_channels,
                time_embed_dim,
                dropout,
                out_channels=out_channels,
                up=upsample,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

    def forward(self, h, embeddings):
        return self.down_block(h, embeddings)
