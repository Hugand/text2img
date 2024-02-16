import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torchvision.utils import save_image
from generative_models.sgm.modules.attention import SpatialTransformer
from typing import Optional

from generative_models.sgm.modules.diffusionmodules.openaimodel import Downsample, ResBlock, TimestepEmbedSequential, Upsample
from generative_models.sgm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding

class DecodingUnet(nn.Module):
    def __init__(self,
        model_channels: int,
        dropout: float = 0.0,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        context_dim: Optional[int] = None,
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

        # In
        self.input = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        
        # Down Block 1
        self.down_block_1 = TimestepEmbedSequential(
            ResBlock(128, time_embed_dim, self.dropout, out_channels=128, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                128,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(128, time_embed_dim, self.dropout, out_channels=256, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Downsample(128, use_conv=self.conv_resample, out_channels=256),
        )
        # self.attention_1 = LatentConditioningCrossAttention(256, context_dim=self.context_dim, heads=8, dim_head=64)

        # Down Block 2
        self.down_block_2 = TimestepEmbedSequential(
            ResBlock(256, time_embed_dim, self.dropout, out_channels=256, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                256,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(256, time_embed_dim, self.dropout, out_channels=512, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Downsample(256, use_conv=self.conv_resample, out_channels=512)
        )
        # self.attention_2 = LatentConditioningCrossAttention(512, context_dim=self.context_dim, heads=8, dim_head=64)

        # Down Block 3
        self.down_block_3 = TimestepEmbedSequential(
            ResBlock(512, time_embed_dim, self.dropout, out_channels=512, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                512,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(512, time_embed_dim, self.dropout, out_channels=512, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Downsample(512, use_conv=self.conv_resample, out_channels=512)
        )

        # Middle Block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(512, time_embed_dim, self.dropout, out_channels=512, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                512,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(512, time_embed_dim, self.dropout, out_channels=512, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
        )

        # Up Block 1
        self.up_block_1 = TimestepEmbedSequential(
            ResBlock(512+512, time_embed_dim, self.dropout, out_channels=512, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                512,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(512, time_embed_dim, self.dropout, out_channels=256, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Upsample(512, out_channels=256, use_conv=self.conv_resample)
        )

        # Up Block 2
        self.up_block_2 = TimestepEmbedSequential(
            ResBlock(256+512, time_embed_dim, self.dropout, out_channels=256, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                256,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(256, time_embed_dim, self.dropout, out_channels=128, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Upsample(256, out_channels=128, use_conv=self.conv_resample)
        )
        # self.attention_3 = LatentConditioningCrossAttention(128, context_dim=self.context_dim, heads=8, dim_head=64)

        # Up Block 3
        self.up_block_3 = TimestepEmbedSequential(
            ResBlock(128+256, time_embed_dim, self.dropout, out_channels=128, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            SpatialTransformer(
                128,
                8,
                32,
                depth=1,
                context_dim=4,
                disable_self_attn=True,
                use_linear=True,
                attn_type="softmax-xformers",
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(128, time_embed_dim, self.dropout, out_channels=128, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
            if self.resblock_updown
            else Upsample(128, out_channels=128, use_conv=self.conv_resample)
        )
        # self.attention_4 = LatentConditioningCrossAttention(128, context_dim=self.context_dim, heads=8, dim_head=64)

        # Out
        self.out_block = TimestepEmbedSequential(
            GroupNorm32(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, 
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ):
        context_reshaped = rearrange(context, "b c h w -> b (h w) c")
        
        time_embeddings = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        embeddings = self.time_embed(time_embeddings)

        h = self.input(x)
        hs = []

        h = self.down_block_1(h, embeddings, context=context_reshaped)
        # h = self.attention_1(h, context)
        hs.append(h)

        h = self.down_block_2(h, embeddings, context=context_reshaped)
        # h = self.attention_2(h, context)
        hs.append(h)

        h = self.down_block_3(h, embeddings, context=context_reshaped)
        hs.append(h)

        h = self.middle_block(h, embeddings, context=context_reshaped)
    
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_1(h, embeddings, context=context_reshaped)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_2(h,  embeddings, context=context_reshaped)
        # h = self.attention_3(h, context)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_3(h,  embeddings, context=context_reshaped)
        # h = self.attention_4(h, context)

        h = self.out_block(h, embeddings)

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

        unet = DecodingUnet().to(device)
        out = unet(noise, timesteps, latent)

        images = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

        print("# Saving image...")
        save_image(images[0], 'decoded_unet_img.png')

