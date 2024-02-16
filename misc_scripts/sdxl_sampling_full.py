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

def build_value_dict(params, prompt, negative_prompt):
    value_dict = asdict(params)
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = negative_prompt
    value_dict["orig_width"] = params.width
    value_dict["orig_height"] = params.height
    value_dict["target_width"] = params.width
    value_dict["target_height"] = params.height

    return value_dict

def generate_img(prompt):
    # Load VAE
    model_params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    base = SamplingPipeline(model_params[0], config_path="generative_models/configs/inference")
    refiner = SamplingPipeline(model_params[1], config_path="generative_models/configs/inference")

    n_samples = 1

    sampler_enum = api.Sampler.EULER_EDM
    params = SamplingParams(sampler=sampler_enum.value, steps=10)
    negative_prompt=""
    H = params.height
    W = params.width
    C = base.specs.channels
    F = base.specs.factor

    steps = 40
    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
        "params": {
            "sigma_min": 0.03,
            "sigma_max": 14.61,
            "rho": 3.0,
        },
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
            with base.model.ema_scope():
                n_samples = [n_samples]
                value_dict = build_value_dict(params, prompt, negative_prompt)
                
                load_model(base.model.conditioner)
                
                batch, batch_uc = helpers.get_batch(
                    helpers.get_unique_embedder_keys_from_conditioner(base.model.conditioner), 
                    value_dict,
                    n_samples,
                )

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])

                force_uc_zero_embeddings=["txt"]
                force_cond_zero_embeddings=[]
                batch2model_input = []

                # Get the conditional and unconditional text embeddings
                c, uc = base.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_model(base.model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(n_samples)].to(device), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                # Shape of the latent
                shape = (math.prod(n_samples), C, H // F, W // F)
                # Noise latent: x_T
                randn = torch.randn(shape).to(device)

                def denoiser(input, sigma, c):
                    return base.model.denoiser(
                        base.model.model, input, sigma, c, **additional_model_inputs
                    )
                
                # Apply the reverse process
                load_model(base.model.denoiser)
                load_model(base.model.model)
                latent_x0 = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(base.model.model)
                unload_model(base.model.denoiser)

                # Apply refiner
                value_dict = {
                    "orig_width": value_dict["orig_width"],
                    "orig_height": value_dict["orig_height"],
                    "target_width": value_dict["target_width"],
                    "target_height": value_dict["target_height"],
                    "prompt": value_dict["prompt"],
                    "negative_prompt": value_dict["negative_prompt"],
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
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_model(refiner.model.conditioner)

                for k in c:
                    c[k], uc[k] = map(
                        lambda y: y[k][: math.prod(n_samples)].to(device), (c, uc)
                    )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

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

                load_model(refiner.model.denoiser)
                load_model(refiner.model.model)
                refined_latent = sampler(denoiser, noised_latent_x0, cond=c, uc=uc)
                unload_model(refiner.model.model)
                unload_model(refiner.model.denoiser)

                # Decode latent with the VAE
                load_model(base.model.first_stage_model)
                base.model.en_and_decode_n_samples_a_time = (
                    1  # Decode n frames at a time
                )
                out_image = base.model.decode_first_stage(refined_latent)
                
                # Map the data from [-1, 1] to [0, 1]
                samples = torch.clamp((out_image + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(base.model.first_stage_model)

                # Save decoded image
                save_image(samples[0], 'generated_1.png')


if __name__ == '__main__':
    generate_img("A photorealistic photo of a Jormungandr coming out of a man's mouth")