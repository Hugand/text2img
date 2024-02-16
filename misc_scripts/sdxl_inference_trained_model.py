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

def build_value_dict(params, conditioning_type, value):
    value_dict = asdict(params)
    
    if conditioning_type == "prompt":
        value_dict["negative_prompt"] = ""
    value_dict[conditioning_type] = value
    value_dict["orig_width"] = params.width
    value_dict["orig_height"] = params.height
    value_dict["target_width"] = params.width
    value_dict["target_height"] = params.height

    return value_dict

def generate_img(conditioning_type, conditioning_value, model, n_samples=1, steps=50):
    conditioning_value = torch.tensor(conditioning_value, dtype=torch.int)
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
            "scale": 5.0,
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

    sampler_enum = api.Sampler.EULER_EDM
    params = SamplingParams(sampler=sampler_enum.value, steps=steps)
    negative_prompt=""
    H = params.height
    W = params.width
    C = 3
    F = 8
    n_samples = [n_samples]

    force_uc_zero_embeddings=["cls"]
    force_cond_zero_embeddings=[]
    batch2model_input = []

    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                value_dict = build_value_dict(params, conditioning_type, conditioning_value)

                embs = helpers.get_unique_embedder_keys_from_conditioner(model.conditioner)

                print(embs)
                
                batch, batch_uc = helpers.get_batch(
                    embs, 
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
                # Get the conditional and unconditional text embeddings
                print(batch_uc)
                print(force_uc_zero_embeddings)
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(n_samples)].to(device), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                # Shape of the latent
                # shape = (math.prod(n_samples), C, H // F, W // F)
                shape = (math.prod(n_samples), C, 64, 64)
                print("SHAPE:", shape)
                # Noise latent: x_T
                randn = torch.randn(shape).to(device)
                print(randn)

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )
                load_model(model.denoiser)
                load_model(model.model)
                # Apply the reverse process
                latent_x_0 = sampler(denoiser, randn, cond=c, uc=uc)
                print(latent_x_0)
                unload_model(model.model)
                unload_model(model.denoiser)

                # torch.save(latent_x_0, "outputs/latents/base/tmp_train_latent3.pt")
                print(latent_x_0.shape)
                # Decode latent with the VAE
                # load_model(model.first_stage_model)
                # out_image = model.decode_first_stage(latent_x_0)
                # unload_model(model.first_stage_model)
                # print(model.first_stage_model)
                # print(out_image)
                
                # Map the data from [-1, 1] to [0, 1]
                # samples = torch.clamp((out_image + 1.0) / 2.0, min=0.0, max=1.0)
                # latent = torch.clamp((latent_x_0 + 1.0) / 2.0, min=0.0, max=1.0)

                return latent_x_0, latent_x_0



model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
model = instantiate_from_config(model_config.model)
model = model.load_from_checkpoint(
    "lightning_logs/version_28/checkpoints/tmp.ckpt",
    network_config=model_config.model.params.network_config,
    first_stage_config=model_config.model.params.first_stage_config,
    denoiser_config=model_config.model.params.denoiser_config,
    conditioner_config=model_config.model.params.conditioner_config,
    loss_fn_config=model_config.model.params.loss_fn_config,
    sampler_config=model_config.model.params.sampler_config,
)
# vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# vae = vae.to("cuda")
# vae.freeze()

# model.first_stage_model = vae

# disable randomness, dropout, etc...
model.eval()
# vae.eval()

print("# Sampling new image....")

images, latent = generate_img("cls", [3], model)

# images = vae.decode(latent)
images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)


print("# Saving image...")
save_image(images[0], 'generated_from_trained3.png')
save_image(latent[0], 'generated_from_trained_l3.png')

