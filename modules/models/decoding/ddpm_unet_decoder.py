import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.model import Downsample

from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential
from sgm.modules.diffusionmodules.util import GroupNorm32, timestep_embedding
# from sgm.modules.attention import SelfAttention
from sgm.modules.diffusionmodules.model import Upsample

from torchvision.utils import save_image

class SelfAttentionBlock(nn.Module):
    """
    Self-attention blocks are applied at the 16x16 resolution in the original DDPM paper.
    Implementation is based on "Attention Is All You Need" paper, Vaswani et al., 2015
    (https://arxiv.org/pdf/1706.03762.pdf)
    """
    def __init__(self, num_heads, in_channels, num_groups=32, embedding_dim=256):
        super(SelfAttentionBlock, self).__init__()
        # For each of heads use d_k = d_v = d_model / num_heads
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        self.d_values = embedding_dim // num_heads

        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)

        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def split_features_for_heads(self, tensor):
        # We receive Q, K and V at shape [batch, h*w, embedding_dim].
        # This method splits embedding_dim into 'num_heads' features so that
        # each channel becomes of size embedding_dim / num_heads.
        # Output shape becomes [batch, num_heads, h*w, embedding_dim/num_heads],
        # where 'embedding_dim/num_heads' is equal to d_k = d_k = d_v = sizes for
        # K, Q and V respectively, according to paper.
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1)
        heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
        return heads_splitted_tensor

    def forward(self, input_tensor):
        x = input_tensor
        batch, features, h, w = x.shape
        # Do reshape and transpose input tensor since we want to process depth feature maps, not spatial maps
        x = x.view(batch, features, h * w).transpose(1, 2)

        # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
        queries = self.query_projection(x)  # [b, in_channels, embedding_dim]
        keys = self.key_projection(x)       # [b, in_channels, embedding_dim]
        values = self.value_projection(x)   # [b, in_channels, embedding_dim]

        # Split Q, K, V between attention heads to process them simultaneously
        queries = self.split_features_for_heads(queries)
        keys = self.split_features_for_heads(keys)
        values = self.split_features_for_heads(values)

        # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
        # Each SDPA block yields tensor of size d_v = embedding_dim/num_heads.
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, values)

        # Permute computed attention scores such that
        # [batch, num_heads, h*w, embedding_dim] --> [batch, h*w, num_heads, d_v]
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # Concatenate scores per head into one tensor so that
        # [batch, h*w, num_heads, d_v] --> [batch, h*w, num_heads*d_v]
        concatenated_heads_attention_scores = attention_scores.view(batch, h * w, self.d_model)

        # Perform linear projection and reshape tensor such that
        # [batch, h*w, d_model] --> [batch, d_model, h*w] -> [batch, d_model, h, w]
        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(batch, self.d_model, h, w)

        # Residual connection + norm
        x = self.norm(linear_projection + input_tensor)
        return x

class TransformerPositionalEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension, device="cuda")
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        # [bs, d_model]
        return self.pe_matrix[timestep].to(timestep.device)

class DDPMUNetDecoder(nn.Module):
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
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            # TransformerPositionalEmbedding(dimension=model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # In - Gets 3 channels of noise + 4 channels of the latent
        self.input = nn.Conv2d(3+4, model_channels, kernel_size=3, stride=1, padding=1)
        
        self.input_blocks = nn.ModuleList([
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
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
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
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                SelfAttentionBlock(in_channels=model_channels * 2, num_heads=4, embedding_dim=model_channels * 2),
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                SelfAttentionBlock(in_channels=model_channels * 2, num_heads=4, embedding_dim=model_channels * 2),
                nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 4,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels * 4,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 4,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
            ),
        ])

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                model_channels * 4,
                time_embed_dim,
                dropout,
                out_channels=model_channels * 4,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SelfAttentionBlock(in_channels=model_channels * 4, num_heads=4, embedding_dim=model_channels * 4),
            ResBlock(
                model_channels * 4,
                time_embed_dim,
                dropout,
                out_channels=model_channels * 4,
                down=False,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SelfAttentionBlock(in_channels=model_channels * 4, num_heads=4, embedding_dim=model_channels * 4),
        )

        self.out_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 4 + model_channels * 4,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 4,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels * 4,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 4,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                Upsample(model_channels * 4, with_conv=True)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 4 + model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                SelfAttentionBlock(in_channels=model_channels * 2, num_heads=4, embedding_dim=model_channels * 2),
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                SelfAttentionBlock(in_channels=model_channels * 2, num_heads=4, embedding_dim=model_channels * 2),
                Upsample(model_channels * 2, with_conv=True)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 2 + model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels * 2,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * 2,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                Upsample(model_channels * 2, with_conv=True)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels * 2 + model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                Upsample(model_channels, with_conv=True)
            ),
            TimestepEmbedSequential(
                ResBlock(
                    model_channels + model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                ResBlock(
                    model_channels,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels,
                    down=False,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                ),
                Upsample(model_channels, with_conv=True)
            ),
        ])

        # Out
        self.output = TimestepEmbedSequential(
            GroupNorm32(32, model_channels * 2),
            nn.SiLU(),
            nn.Conv2d(model_channels * 2, 3, kernel_size=3, stride=1, padding=1)
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

        # h = x
        hs = [h]

        for module in self.input_blocks:
            h = module(h, embeddings)
            hs.append(h)

        h = self.middle_block(h, embeddings)

        for module in self.out_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), embeddings)
        h = torch.cat([h, hs.pop()], dim=1)
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
        unet = DDPMUNetDecoder(128).to(device)
        out = unet(noise, timesteps, latent)

        images = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

        print(images.shape)

        print("# Saving image...")
        save_image(images[0], 'ddpm_decoded_img.png')

