from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from dataloaders.coco17_loader import COCO17Loader
from sgm.util import instantiate_from_config
import torch

model_config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
model = instantiate_from_config(model_config.model)

print(model.first_stage_model)