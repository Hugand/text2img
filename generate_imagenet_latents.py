from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
import torch
from torchvision.utils import save_image
from torch import autocast
import math
from dataclasses import asdict
from einops import rearrange, repeat
from tqdm import tqdm
from dataloaders.coco17_loader import COCO17Loader
from dataloaders.imagenet_loader import ImageNetLatentLoader
import os
from diffusers.models import AutoencoderKL

import generative_models.sgm.inference.helpers as helpers
import generative_models.sgm.inference.api as api
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from generative_models.sgm.util import disabled_train, load_model_from_config

# model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
# model = instantiate_from_config(model_config.model)
# vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# vae = vae.to("cuda")

# images = torch.Tensor([])
# latents = torch.Tensor([])

# Stopped in 10419
loader = ImageNetLatentLoader(
    256, num_workers=10, dims=(128, 128), test_frac=0,
    crop=True, latents_subdir="latents_train_sdxl_128/",
    shuffle=False, random_seed=100
)
# loader = COCO17Loader(
#     32, num_workers=10, test_frac=0, dims=(128, 128), crop=True,
#     latents_subdir="val2017_latents_sdxl_128/",
# )
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, device="cuda:0"
# )
# pipe = StableDiffusionXLPipeline.from_single_file(
#     "checkpoints/sd_xl_base_1.0.safetensors",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sdxl-vae",
).cuda()

# pipe.unet.cpu()
# pipe.vae.cuda()
l = 0
with torch.no_grad():
    for batch in tqdm(loader.train_dataloader()):
        # l += 1
        # if l < 6110:
        #     continue
        latent = vae.encode(batch["jpg"].to("cuda")).latent_dist.mean.cpu()
        print(batch["ltnt"][0])

        for i in range(256):
            latent_dir = "/".join(batch["ltnt"][i].split("/")[:-1]) + "/"
            # print(latent_dir)
            if not os.path.exists(latent_dir):
                os.mkdir(latent_dir)
                
            torch.save(latent[i], batch["ltnt"][i])
