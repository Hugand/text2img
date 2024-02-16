import torch
from diffusers import DiffusionPipeline, ConsistencyDecoderVAE
from PIL import Image
import numpy as np

device = "cuda"

vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16).to(device)

image = Image.open("vae_out.png")
#image = np.asarray(image)

w, h = image.size
print(f"loaded input image of size ({w}, {h})")
image = image.resize((256, 256))
image = np.array(image.convert("RGB"))
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image).to(dtype=torch.float16) / 127.5 - 1.0
image = image.to(device)

print(vae.config.block_out_channels)

# latent = vae.encode(image).latent_dist
# latent = latent.sample()

# print(latent)
# print(latent.shape)

# recon = vae.decode(latent)

# print(recon.shape)
