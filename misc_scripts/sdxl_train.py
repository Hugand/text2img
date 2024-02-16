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
# vae = instantiate_from_config(model_config.first_stage_config)
# vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# vae = vae.to("cuda")
# vae.train = disabled_train
# for param in vae.parameters():
#     param.requires_grad = False

model = instantiate_from_config(model_config.model)
model = model.load_from_checkpoint(
    "lightning_logs/version_26/checkpoints/epoch=199-step=39200.ckpt",
    network_config=model_config.model.params.network_config,
    first_stage_config=model_config.model.params.first_stage_config,
    denoiser_config=model_config.model.params.denoiser_config,
    conditioner_config=model_config.model.params.conditioner_config,
    loss_fn_config=model_config.model.params.loss_fn_config,
    sampler_config=model_config.model.params.sampler_config,
)
model.learning_rate = model_config.model.base_learning_rate
# model.first_stage_model = vae

trainer = Trainer(**trainer_opt)
trainer.fit(model, data)
