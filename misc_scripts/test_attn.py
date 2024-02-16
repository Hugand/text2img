
from einops import rearrange
import torch

from generative_models.sgm.modules.attention import MemoryEfficientCrossAttention, SpatialSelfAttention, SpatialTransformer
from generative_models.sgm.modules.diffusionmodules.model import MemoryEfficientCrossAttentionWrapper




class LatentConditioningCrossAttention(MemoryEfficientCrossAttention):
    def forward(self, x, latent=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        latent = rearrange(latent, "b c h w -> b (h w) c")
        out = super().forward(x, context=latent)
        x = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        print("X:", x.shape)
        print("Out:", out.shape)
        return x + out


# # spatial_attention = SpatialSelfAttention(64).to(device)
# attention = MemoryEfficientCrossAttention(
#     query_dim=64,
#     context_dim=4,
#     heads=8,
#     dim_head=64
# ).to(device)

# b, c, h, w = noise.shape
# x = rearrange(noise, "b c h w -> b (h w) c")
# context = rearrange(latent, "b c h w -> b (h w) c")
# out = attention.forward(x, context=context, mask=None)
# out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
# attention = LatentConditioningCrossAttention(256, context_dim=4, heads=8, dim_head=64).to(device)

# 128 /  64
attention = SpatialTransformer(
    128,
    8,
    32,
    depth=1,
    context_dim=4,
    disable_self_attn=True,
    use_linear=True,
    attn_type="softmax-xformers",
    use_checkpoint=True,
).to("cuda")

# MemoryEfficientCrossAttention
# attention = MemoryEfficientCrossAttention(
#     query_dim=128,
#     heads=8,
#     dim_head=32,
#     dropout=0.0,
#     context_dim=4
# ).to("cuda")

noise = torch.randn((1, 128, 32, 32)).to("cuda")
latent = torch.randn((1, 4, 4, 4)).to("cuda")
# x = rearrange(noise, "b c h w -> b (h w) c")
l = rearrange(latent, "b c h w -> b (h w) c")
# print(x.shape)


# out = ( attention(x, l) + x )
out = attention(noise, l)

print("X:", noise.shape)
print("Out:", out.shape)

print(out.shape)