import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image
from typing import Optional

from generative_models.sgm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from generative_models.sgm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding

from modules.blocks.unet_blocks import DecodingUNetDownBlock, DecodingUNetUpBlock

class DecodingUnetConcat(nn.Module):
    def __init__(self,
        model_channels: int,
        dropout: float = 0.0,
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

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # In - Gets 3 channels of noise + 4 channels of the latent
        self.input = nn.Conv2d(3+4, model_channels, kernel_size=3, stride=1, padding=1)

        self.down_block_1 = DecodingUNetDownBlock(
                model_channels,
                model_channels*2,
                time_embed_dim,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            )
        self.down_block_2 = DecodingUNetDownBlock(
                model_channels*2,
                model_channels*4,
                time_embed_dim,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            )
        self.down_block_3 = DecodingUNetDownBlock(
                model_channels*4,
                model_channels*4,
                time_embed_dim,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            )
        self.down_block_4 = DecodingUNetDownBlock(
                model_channels*4,
                model_channels*4,
                time_embed_dim,
                self.dropout,
                downsample=False,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
            )

        self.middle_block = DecodingUNetDownBlock(
                model_channels*4,
                model_channels*4,
                time_embed_dim,
                self.dropout,
                downsample=False,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
        )

        self.up_block_1 = DecodingUNetUpBlock(
                model_channels*4+model_channels*4,
                model_channels*4,
                model_channels*4,
                time_embed_dim,
                upsample=False,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
        )

        self.up_block_2 = DecodingUNetUpBlock(
                model_channels*4+model_channels*4,
                model_channels*4,
                model_channels*2,
                time_embed_dim,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
        )

        self.up_block_3 = DecodingUNetUpBlock(
                model_channels*2+model_channels*4,
                model_channels*2,
                model_channels,
                time_embed_dim,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
        )

        self.up_block_4 = DecodingUNetUpBlock(
                model_channels+model_channels*2,
                model_channels,
                model_channels,
                time_embed_dim,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm
        )

        # Out
        self.out_block = TimestepEmbedSequential(
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
        print(x.shape, context.shape)
        upscale_factor = 2 ** (4-1)
        z = F.interpolate(context, mode="nearest", scale_factor=upscale_factor)
        print(x.shape, z.shape)
        h = torch.cat([x, z], dim=1)

        time_embeddings = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        embeddings = self.time_embed(time_embeddings)

        h = self.input(h)
        hs = []
    
        print("in", h.shape)
        h = self.down_block_1(h, embeddings)
        hs.append(h)

        print("d1", h.shape)
        h = self.down_block_2(h, embeddings)
        hs.append(h)

        print("d2", h.shape)
        h = self.down_block_3(h, embeddings)
        hs.append(h)

        print("d3", h.shape)
        h = self.down_block_4(h, embeddings)
        hs.append(h)

        print("d4", h.shape)
        h = self.middle_block(h, embeddings)

        print("m", h.shape)
        hp = hs.pop()
        h = torch.cat([h, hp], dim=1)
        h = self.up_block_1(h, embeddings)

        print("u1", h.shape)
        hp = hs.pop()
        h = torch.cat([h, hp], dim=1)
        h = self.up_block_2(h,  embeddings)

        print("u2", h.shape)
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_3(h,  embeddings)

        print("u3", h.shape)
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_4(h,  embeddings)

        print("u4", h.shape)
        h = self.out_block(h, embeddings)
        print(h.shape)

        return h


if __name__ == "__main__":
    device = "cuda"

    noise_shape = (1, 3, 32, 32)
    latent_shape = (1, 4, 4, 4)

    torch.cuda.empty_cache()

    with torch.no_grad():

        noise = torch.randn(noise_shape).to(device)
        latent = torch.randn(latent_shape).to(device)
        timesteps = torch.Tensor([2]).to(device)

        # unet = DecodingUnet().to(device)
        unet = DecodingUnetConcat(320).to(device)
        out = unet(noise, timesteps, latent)

        images = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

        print("# Saving image...")
        save_image(images[0], 'decoded_unet_img.png')

