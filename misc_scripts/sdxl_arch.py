from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image

from generative_models.sgm.inference.api import (
    SamplingPipeline,
    SamplingSpec,
    ModelArchitecture,
)
import generative_models.sgm.inference.helpers as helpers

device = "cuda"

model_specs = {
    ModelArchitecture.SDXL_V1_BASE: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=False,
        config="sd_xl_base.yaml",
        ckpt="sd_xl_base_1.0.safetensors",
        is_guided=True,
    ),
    ModelArchitecture.SDXL_V1_REFINER: SamplingSpec(
        height=1024,
        width=1024,
        channels=4,
        factor=8,
        is_legacy=True,
        config="sd_xl_refiner.yaml",
        ckpt="sd_xl_refiner_1.0.safetensors",
        is_guided=True,
    )
}

def print_dm_arch():
    params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    base_pipeline = SamplingPipeline(params[0], config_path="generative_models/configs/inference")

    print(base_pipeline.model.model)

if __name__ == '__main__':
    print_dm_arch()
