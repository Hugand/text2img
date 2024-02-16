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

from generative_models.sgm.inference.api import (
    SamplingParams,
    SamplingPipeline,
    SamplingSpec,
    ModelArchitecture,
)
import generative_models.sgm.inference.helpers as helpers
import generative_models.sgm.inference.api as api
from sgm.modules.diffusionmodules.guiders import LinearPredictionGuider, VanillaCFG
from sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from sgm.util import append_dims, instantiate_from_config

from generative_models.sgm.util import disabled_train, load_model_from_config


trainer_opt = {
    "benchmark": True,
    "num_sanity_val_steps": 0,
    "accumulate_grad_batches": 1,
    "max_epochs": 1000,
    "accelerator": "cuda"
}

dataset_config = OmegaConf.load("configs/data/cifar10.yaml")
print(dataset_config)
data = instantiate_from_config(dataset_config.data)
print(data)

data.prepare_data()
data.setup("dummy")
print("#### Data #####")
try:
    for k in data.datasets:
        print(
            f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
        )
except:
    print("datasets not yet initialized.")

print(data)


model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
model = instantiate_from_config(model_config.model)
vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
vae = vae.to("cuda")

images = torch.Tensor([])
latents = torch.Tensor([])

with torch.no_grad():
    for batch in data.train_dataloader():
        latent_batch = vae.encode(batch["jpg"].to("cuda"))

        latents = torch.cat([latents, latent_batch.to("cpu")], dim=0)
        images = torch.cat([images, batch["jpg"].to("cpu")], dim=0)

    images = torch.Tensor(images)
    latents = torch.Tensor(latents)

    torch.save(images, "./datasets/cifar10-latents/cifar10-imgs.pt")
    torch.save(latents, "./datasets/cifar10-latents/cifar10-latents.pt")

    print(images.shape, latents.shape)


# model_config = OmegaConf.load("configs/models/cifar10_model.yaml")

# trainer = Trainer(**trainer_opt)
# trainer.fit(model, data)

# Load VAE
# Encode an image dataset to the latent space
# Save (latent, image) pairs dataset
# Delete VAE to save memory

# Instantiate a Diffusion Model from Ho et al.
# - Smaller than the SDXL U-Net
# - Probably original Ho et al. U-Net
# - Implement a conditioning mechanism for the latent space
# Train the Diffusion Model to generate images from latents
# Apply Consistency and Adversarial Diffusion Distillation methods
# Compute the metrics and compare new VAE vs old VAE
# Compare distillation techniques
