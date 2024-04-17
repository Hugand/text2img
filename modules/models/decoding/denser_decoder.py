from typing import Optional
import torch.nn.functional as F
from sgm.modules.diffusionmodules.util import timestep_embedding
import torch.nn as nn
from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential
import torch


# Put all of the blocks together in the model
class DenserDiffusionDecoder(nn.Module):
    def __init__(self,
        model_channels: int,
        dropout: float = 0.0,
        input_channels: int = 4,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        scaling_factor: float = 0.18215
    ):
        super().__init__()
        self.model_channels = model_channels
        self.input_channels = input_channels
        self.input = nn.Conv2d(input_channels, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.time_embed = nn.Sequential(
            nn.Linear(in_features=320, out_features=1280, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=1280, out_features=1280, bias=True)
        )
        self.down_blocks = nn.ModuleList([
            # 1
            TimestepEmbedSequential(ResBlock(
                320, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                320, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                320, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                320, 1280, dropout, out_channels=320, down=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 2,
            TimestepEmbedSequential(ResBlock(
                320, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=640, down=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 3
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024, down=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 4
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1024, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            ))
        ])
        self.middle_block = TimestepEmbedSequential(ResBlock(
            1024, 1280, dropout, out_channels=1024,
            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
        ), ResBlock(
            1024, 1280, dropout, out_channels=1024,
            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
        ))
        self.up_blocks = nn.ModuleList([
            # 3
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            ), ResBlock(
                1024, 1280, dropout, out_channels=1024, up=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 2
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                2048, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1664, 1280, dropout, out_channels=1024,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            ), ResBlock(
                1024, 1280, dropout, out_channels=1024, up=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 1
            TimestepEmbedSequential(ResBlock(
                1664, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1280, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                1280, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                960, 1280, dropout, out_channels=640,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            ), ResBlock(
                640, 1280, dropout, out_channels=640, up=True,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),

            # 0
            TimestepEmbedSequential(ResBlock(
                960, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            )),
            TimestepEmbedSequential(ResBlock(
                640, 1280, dropout, out_channels=320,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
            ))
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(32, 320, eps=1e-05, affine=True),
            nn.SiLU(),
            nn.Conv2d(320, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        self.latent_scaler = nn.ModuleList([
            ResBlock(4, 0, 0.0, 64, skip_t_emb=True, use_checkpoint=use_checkpoint),
            ResBlock(64, 0, 0.0, 128, skip_t_emb=True, use_checkpoint=use_checkpoint),
            ResBlock(256, 0, 0.0, 256, skip_t_emb=True, use_checkpoint=use_checkpoint),
            ResBlock(256, 0, 0.0, 256, skip_t_emb=True, use_checkpoint=use_checkpoint),
            ResBlock(256, 0, 0.0, 7, skip_t_emb=True, use_checkpoint=use_checkpoint)
        ])

    def from_cdd(self, cdd):
        # Loads every layer except the input layer
        self.time_embed = cdd.time_embed
        self.down_blocks = cdd.down_blocks
        self.middle_block = cdd.middle_block
        self.up_blocks = cdd.up_blocks
        self.output = cdd.output

    def forward(self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ):
        # z = context
        # Upscale latent
        z = self.latent_scaler(context)
        upscale_factor = 2 ** (4-1)
        z = F.interpolate(z, mode="nearest", scale_factor=upscale_factor)
        h = torch.cat([x, z], dim=1)

        time_embeddings = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        embeddings = self.time_embed(time_embeddings)

        h = self.input(h)

        # h = x
        hs = [h]

        for module in self.down_blocks:
            h = module(h, embeddings)
            hs.append(h)

        h = self.middle_block(h, embeddings)

        for module in self.up_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), embeddings)

        out = self.output(h)

        B, C = x.shape[:2]
        model_output, _ = torch.split(out, C, dim=1)
        return model_output