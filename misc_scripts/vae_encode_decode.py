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

def print_vae_arch():
    params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    base_pipeline = SamplingPipeline(params[0], config_path="generative_models/configs/inference")

    print(base_pipeline.model.first_stage_model)

def vae_encode(image_path, save_latent=False, decode_latent=False, latent_path=None):
    # Load VAE
    params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    base = SamplingPipeline(params[0], config_path="generative_models/configs/inference")
    vae = base.model.first_stage_model
    
    if not decode_latent:
        image = Image.open(image_path)

        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        width, height = map(
            lambda x: x - x % 64, (w, h)
        )  # resize to integer multiple of 64
        image = image.resize((width, height))
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        image = image.to(device)


        # Encode image
        latent = vae.encode(image)

        if save_latent:
            torch.save(latent, "latent_x0.pt")
    else:
        if not latent_path == None:
            latent = torch.load(latent_path)
        else: 
            latent = torch.load("refined.pt")

    # Decode image
    out_image = vae.decode(latent)
    out_image = torch.clamp((out_image + 1.0) / 2.0, min=0.0, max=1.0)


    # Save decoded image
    save_image(out_image[0], 'vae_out.png')

if __name__ == '__main__':
    vae_encode("generated_from_training_decoded.png", save_latent=False, decode_latent=True, latent_path='outputs/latent/base/tmp_train_latent.pt')
