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
from dataloaders.imagenet_loader import ImageNetLatentLoader
import os

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
from torchvision.datasets import ImageNet



model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
model = instantiate_from_config(model_config.model)
vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
vae = vae.to("cuda")

# images = torch.Tensor([])
# latents = torch.Tensor([])
imagenet_loader = ImageNetLatentLoader(1, num_workers=10, dims=(128, 128), test_frac=0)

with torch.no_grad():
    for batch in tqdm(imagenet_loader.train_dataloader()):
        latent = vae.encode(batch["jpg"].to("cuda")).to("cpu")

        # for i in range(latent.shape[0]):
        #     # print(batch["ltnt"])
        latent_dir = "/".join(batch["ltnt"][0].split("/")[:-1]) + "/"
        if not os.path.exists(latent_dir):
            os.mkdir(latent_dir)
            
        torch.save(latent[0], batch["ltnt"][0])
