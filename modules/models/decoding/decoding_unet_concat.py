import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.model import Downsample

from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential
from sgm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding

from modules.models.decoding.unet_blocks import DecodingUNetDownBlock, DecodingUNetUpBlock
from torchvision.utils import save_image


class DecodingUnetConcat(nn.Module):
    def __init__(self,
        model_channels: int,
        dropout: float = 0.0,
        num_head_channels: int = 32,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        context_dim: Optional[int] = None,
        scaling_factor: float = 0.18215
    ):
        super().__init__()
        time_embed_dim = model_channels

        self.dropout = dropout
        self.model_channels = model_channels
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.context_dim = context_dim
        self.num_head_channels = num_head_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # In - Gets 3 channels of noise + 4 channels of the latent
        self.input = TimestepEmbedSequential(
            nn.Conv2d(3+4, model_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.input_blocks = nn.ModuleList([
            self.input,
            TimestepEmbedSequential(
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
            ), TimestepEmbedSequential(
                Downsample(
                    in_channels=model_channels,
                    with_conv=True
                )
            ),

            DecodingUNetDownBlock(
                model_channels,
                model_channels*2,
                time_embed_dim,
                num_head_channels,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetDownBlock(
                model_channels*2,
                model_channels*2,
                time_embed_dim,
                num_head_channels,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ), TimestepEmbedSequential(
                Downsample(
                    in_channels=model_channels*2,
                    with_conv=True
                ),
            ),

            DecodingUNetDownBlock(
                model_channels*2,
                model_channels*4,
                time_embed_dim,
                num_head_channels,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetDownBlock(
                model_channels*4,
                model_channels*4,
                time_embed_dim,
                num_head_channels,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
        ])

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                model_channels*4,
                time_embed_dim,
                dropout,
                out_channels=model_channels*4,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialTransformer(
                model_channels*4,
                (model_channels*4) // num_head_channels,
                num_head_channels,
                use_checkpoint=use_checkpoint,
                use_linear=True
            ),
            ResBlock(
                model_channels*4,
                time_embed_dim,
                dropout,
                out_channels=model_channels*4,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

        self.out_blocks = nn.ModuleList([
            DecodingUNetUpBlock(
                model_channels*4+model_channels*4,
                model_channels*4,
                time_embed_dim,
                num_head_channels,
                upsample=False,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetUpBlock(
                model_channels*4+model_channels*4,
                model_channels*4,
                time_embed_dim,
                num_head_channels,
                upsample=False,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetUpBlock(
                model_channels*4+model_channels*2,
                model_channels*4,
                time_embed_dim,
                num_head_channels,
                upsample=True,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),

            DecodingUNetUpBlock(
                model_channels*4+model_channels*2,
                model_channels*2,
                time_embed_dim,
                num_head_channels,
                upsample=False,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetUpBlock(
                model_channels*2+model_channels*2,
                model_channels*2,
                time_embed_dim,
                num_head_channels,
                upsample=False,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            DecodingUNetUpBlock(
                model_channels*2+model_channels,
                model_channels*2,
                time_embed_dim,
                num_head_channels,
                upsample=True,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),

            TimestepEmbedSequential(
                ResBlock(
                    model_channels*2+model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                )
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels+model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                )
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels+model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                )
            ),
        ])

        # Out
        self.output = TimestepEmbedSequential(
            GroupNorm32(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, 
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ):
        # Upscale latent
        upscale_factor = 2 ** (4-1)
        z = F.interpolate(context, mode="nearest", scale_factor=upscale_factor)
        h = torch.cat([x, z], dim=1)

        time_embeddings = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        embeddings = self.time_embed(time_embeddings)

        # h = x
        hs = []

        for module in self.input_blocks:
            # print(module)
            h = module(h, embeddings)
            hs.append(h)
        h = self.middle_block(h, embeddings)

        for module in self.out_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), embeddings)

        return self.output(h, embeddings)


if __name__ == "__main__":
    device = "cuda"

    noise_shape = (1, 3, 256, 256)
    latent_shape = (1, 4, 32, 32)

    torch.cuda.empty_cache()

    with torch.no_grad():

        noise = torch.randn(noise_shape).to(device)
        latent = torch.randn(latent_shape).to(device)
        timesteps = torch.Tensor([2]).to(device)

        # unet = DecodingUnet().to(device)
        unet = DecodingUnetConcat(256).to(device)
        out = unet(noise, timesteps, latent)

        images = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

        print(images.shape)

        print("# Saving image...")
        save_image(images[0], 'decoded_unet_img.png')

