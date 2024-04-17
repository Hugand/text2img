from contextlib import nullcontext
import numpy as np
from omegaconf import ListConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image
from generative_models.sgm.modules.attention import MemoryEfficientCrossAttention, SpatialTransformer
from typing import Iterable, List, Optional, Tuple, Union, Dict

from generative_models.sgm.modules.diffusionmodules.openaimodel import Downsample, ResBlock, TimestepEmbedSequential, Upsample
from generative_models.sgm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding
from sgm.modules.encoders.modules import AbstractEmbModel
from generative_models.sgm.util import count_params, disabled_train, expand_dims_like, instantiate_from_config


class LatentConditioner(nn.Module):
    def __init__(self, emb_models: Union[List, ListConfig], scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch
    
    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = "crossattn"
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), 2
                    )
                else:
                    output[out_key] = emb

                output[out_key] = output[out_key] * self.scale_factor

        return output
    
    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class LatentConditioningCrossAttention(MemoryEfficientCrossAttention):
    def forward(self, x, latent=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        latent = rearrange(latent, "b c h w -> b (h w) c")
        out = super().forward(x, context=latent)
        x = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out

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
        
        # Down Block 1
        self.down_block_1 = TimestepEmbedSequential(
            ResBlock(model_channels, time_embed_dim, self.dropout, out_channels=model_channels, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels, time_embed_dim, self.dropout, out_channels=model_channels*2, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
        )
        self.down_block_2 = TimestepEmbedSequential(
            ResBlock(model_channels*2, time_embed_dim, self.dropout, out_channels=model_channels*2, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*2, time_embed_dim, self.dropout, out_channels=model_channels*4, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
        )
        self.down_block_3 = TimestepEmbedSequential(
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
        )
        self.down_block_4 = TimestepEmbedSequential(
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm)
        )
        # Middle Block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, down=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
        )

        # Up Block 1
        self.up_block_1 = TimestepEmbedSequential(
            ResBlock(model_channels*4+model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
        )
        self.up_block_2 = TimestepEmbedSequential(
            ResBlock(model_channels*4+model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*4, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*2, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
        )
        self.up_block_3 = TimestepEmbedSequential(
            ResBlock(model_channels*2+model_channels*4, time_embed_dim, self.dropout, out_channels=model_channels*2, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*2, time_embed_dim, self.dropout, out_channels=model_channels*2, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels*2, time_embed_dim, self.dropout, out_channels=model_channels, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
        )
        self.up_block_4 = TimestepEmbedSequential(
            ResBlock(model_channels+model_channels*2, time_embed_dim, self.dropout, out_channels=model_channels, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels, time_embed_dim, self.dropout, out_channels=model_channels, up=False, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(model_channels, time_embed_dim, self.dropout, out_channels=model_channels, up=True, use_checkpoint=self.use_checkpoint, use_scale_shift_norm=self.use_scale_shift_norm),
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
        upscale_factor = 2 ** (4-1)
        z = F.interpolate(context, mode="nearest", scale_factor=upscale_factor)

        h = torch.cat([x, z], dim=1)

        time_embeddings = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        embeddings = self.time_embed(time_embeddings)

        h = self.input(h)
        hs = []

        h = self.down_block_1(h, embeddings)
        # h = self.attention_1(h, context)
        hs.append(h)

        h = self.down_block_2(h, embeddings)
        # h = self.attention_2(h, context)
        hs.append(h)

        h = self.down_block_3(h, embeddings)
        hs.append(h)

        h = self.down_block_4(h, embeddings)
        hs.append(h)

        h = self.middle_block(h, embeddings)

        hp = hs.pop()
        h = torch.cat([h, hp], dim=1)
        h = self.up_block_1(h, embeddings)

        hp = hs.pop()
        h = torch.cat([h, hp], dim=1)
        h = self.up_block_2(h,  embeddings)
        # h = self.attention_3(h, context)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_3(h,  embeddings)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block_4(h,  embeddings)
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

        # unet = DecodingUnet().to(device)
        unet = DecodingUnetConcat(320).to(device)
        out = unet(noise, timesteps, latent)

        images = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

        print("# Saving image...")
        save_image(images[0], 'decoded_unet_img.png')

