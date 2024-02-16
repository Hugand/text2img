from PIL import Image
import numpy as np
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
from generative_models.sgm.modules.diffusionmodules.guiders import LinearPredictionGuider, VanillaCFG
from generative_models.sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from generative_models.sgm.util import append_dims

device = "cuda"
lowvram_mode = False


def load_model(model):
    model.cuda()

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()

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

def get_latent_from_image(image_path, vae):
    # Open image
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

    latent = vae.encode(image)

    return latent

def apply_refiner(prompt):
    params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    #vae = SamplingPipeline(ModelArchitecture.SDXL_V1_BASE,  config_path="generative_models/configs/inference")
    #vae = vae.model.first_stage_model
    refiner = SamplingPipeline(ModelArchitecture.SDXL_V1_REFINER,  config_path="generative_models/configs/inference")

    n_samples = 1
    steps = 20

    sampler_enum = api.Sampler.EULER_EDM
    params = SamplingParams(sampler=sampler_enum.value, steps=steps)
    negative_prompt=""
    H = params.height
    W = params.width
    C = refiner.specs.channels
    F = refiner.specs.factor

    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        # "params": {
        #     "sigma_min": 0.03,
        #     "sigma_max": 14.61,
        #     "rho": 3.0,
        # },
    }
    guider_config = {
        "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {
            "scale": 10.0,
        },
    }
    sampler = EulerEDMSampler(
        num_steps=steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=0,
        s_tmin=0,
        s_tmax=999,
        s_noise=1,
        verbose=True,
    )

    with torch.no_grad():
        with autocast(device) as precision_scope:
            with refiner.model.ema_scope():
                n_samples = [n_samples]

                # Apply refiner
                value_dict = {
                    "orig_width": W,
                    "orig_height": H,
                    "target_width": W,
                    "target_height": H,
                    "prompt": prompt,
                    "negative_prompt": "",
                    "crop_coords_top": 0,
                    "crop_coords_left": 0,
                    "aesthetic_score": 6.0,
                    "negative_aesthetic_score": 2.5,
                }
                offset_noise_level = 0.0

                batch, batch_uc = helpers.get_batch(
                    helpers.get_unique_embedder_keys_from_conditioner(refiner.model.conditioner), 
                    value_dict,
                    n_samples,
                )

                force_uc_zero_embeddings = []

                # Get the conditional and unconditional text embeddings
                c, uc = refiner.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                # unload_model(refiner.model.conditioner)

                for k in c:
                    c[k], uc[k] = map(
                        lambda y: y[k][: n_samples[0]].to(device), (c, uc)
                    )

                additional_kwargs = {}
                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]

                # load_model(vae)
                # latent_x0 = get_latent_from_image("generated_1.4.png", vae)
                latent_x0 = torch.load("latent_x0.pt")
                print(latent_x0)
                # torch.save(latent_x0, "latent_x0.pt")
                # unload_model(vae)

                noise = torch.randn_like(latent_x0)
                sigmas = sampler.discretization(sampler.num_steps)
                sigma = sigmas[0].to(latent_x0.device)

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(latent_x0.shape[0], device=latent_x0.device), latent_x0.ndim
                    )
                noised_latent_x0 = latent_x0 + noise * append_dims(sigma, latent_x0.ndim)
                noised_latent_x0 = noised_latent_x0 / torch.sqrt(
                    1.0 + sigmas[0] ** 2.0
                )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

                def denoiser(x, sigma, c):
                    return refiner.model.denoiser(refiner.model.model, x, sigma, c)

                refined_latent = sampler(denoiser, noised_latent_x0, cond=c, uc=uc)
                torch.save(refined_latent, 'refined.pt')

                # Decode latent with the VAE
                out_image = refiner.model.first_stage_model.decode(refined_latent)
                # Map the data from [-1, 1] to [0, 1]
                samples = torch.clamp((out_image + 1.0) / 2.0, min=0.0, max=1.0)
                print("===>", refined_latent)
                # Save decoded image
                save_image(samples, 'refined_1.png')


    print(refiner)

if __name__ == '__main__':
    apply_refiner("A photorealistic photo of a Jormungandr coming out of a man's mouth")
