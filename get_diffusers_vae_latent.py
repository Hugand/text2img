from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from generative_models.sgm.util import load_model_from_config

device = "cuda"

# === Diffusers package ===
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae",
    torch_dtype=torch.float32)
vae.to("cpu")

# === SGM module ===
# model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
# vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# vae = vae.to("cuda")
# vae.eval()

# === Load image ===
image = Image.open("datasets/imagenet/train/n01440764/n01440764_18.JPEG")
smallest_side = min(image.size)
image = image.crop(((
    (image.size[0] - smallest_side) / 2,
    (image.size[1] - smallest_side) / 2,
    (image.size[0] + smallest_side) / 2,
    (image.size[1] + smallest_side) / 2,
)))
image = image.resize((256, 256))
image = np.array(image.convert("RGB"))
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image) / 127.5 - 1.0
image = image.to("cpu")

with torch.no_grad():
    print("# Encoding...")
    latent = vae.encode(image)

    print("# Sampling latent...")
    print(latent)
    # print()
    l = latent.latent_dist.sample()
    print("==>", l.shape)

    torch.save(l, "imagenet_0_latent_diffusers256.pt")
    

    # print("# Decoding...")
    # out = vae.decode(latent)

    # print(out)

# save_image(out, "decoded.png")
