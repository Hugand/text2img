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

from generative_models.sgm.util import load_model_from_config

device = "cuda"
lowvram_mode = False

def load_model(model):
    model.cuda()

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()



model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
vae = vae.to("cpu")
vae.eval()
latent = torch.load("outputs/latents/base/tmp_train_latent.pt").to("cpu")
samples = vae.decode(latent)
samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

save_image(samples[0], 'generated_from_trained2.png')

