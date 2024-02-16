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
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional.image import structural_similarity_index_measure
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

device = "cuda"
seed = 42
torch.manual_seed(seed)

trainer_opt = {
    "benchmark": True,
    "num_sanity_val_steps": 0,
    "accumulate_grad_batches": 1,
    "max_epochs": 400,
    "accelerator": "cuda"
}


dataset = CIFAR10LatentLoader(256, 2)
model_config = OmegaConf.load("configs/models/vae_decoder_concat.yaml")
vae = instantiate_from_config(model_config.model)
vae = vae.load_from_checkpoint(
    "lightning_logs/version_73/checkpoints/tmp.ckpt",
    network_config=model_config.model.params.network_config,
    denoiser_config=model_config.model.params.denoiser_config,
    conditioner_config=model_config.model.params.conditioner_config,
    first_stage_config=model_config.model.params.first_stage_config,
    loss_fn_config=model_config.model.params.loss_fn_config,
    sampler_config=model_config.model.params.sampler_config,
)
vae = vae.to(device)
vae.learning_rate = model_config.model.base_learning_rate


with torch.no_grad():
    for dl_batch in dataset.train_dataloader():
        steps = 50
        n_samples = 24
        print( dl_batch["ltnt"].shape)

        images = dl_batch["jpg"][:n_samples].to("cuda")
        latent = dl_batch["ltnt"][:n_samples].to("cuda")
        images[0] = images[1]
        latent[0] = latent[1]
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
        }
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": 3.0,
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

        force_uc_zero_embeddings=["ltnt"]
        force_cond_zero_embeddings=[]
        batch2model_input = []

        with torch.no_grad():
            with autocast(device) as precision_scope:
                with vae.ema_scope():
                    value_dict = {
                        "ltnt": latent
                    }

                    embs = helpers.get_unique_embedder_keys_from_conditioner(vae.conditioner)
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
                    c, uc = vae.conditioner.get_unconditional_conditioning(
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
                    shape = (math.prod(n_samples), C, 32, 32)
                    randn = torch.randn(shape).to(device)
                    
                    def denoiser(input, sigma, c):
                        return vae.denoiser(
                            vae.model, input, sigma, c, **additional_model_inputs
                        )
                    # Apply the reverse process
                    samples = sampler(denoiser, randn, cond=c, uc=uc)
                    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

                    print("# Calculating LPIPS...")
                    lpips = learned_perceptual_image_patch_similarity(samples, images, net_type='squeeze')
                    psnr = PeakSignalNoiseRatio().to("cuda")
                    psnr_score = psnr(samples, images)
                    ssim = structural_similarity_index_measure(samples, images)


                    # model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
                    # vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
                    # vae = vae.to("cuda")

                    # latents_vae = vae.encode(images)
                    # samples_vae = vae.decode(latents_vae)
                    # samples_vae = torch.clamp((samples_vae + 1.0) / 2.0, min=0.0, max=1.0)
                    # print(latents_vae)
                    # lpips_vae = learned_perceptual_image_patch_similarity(samples_vae, images, net_type='vgg')
                    # psnr_score_vae = psnr(samples_vae, images)
                    # ssim_vae = structural_similarity_index_measure(samples_vae, images)

                    print("LPIPS:", lpips)
                    print("PSNR:", psnr_score)
                    print("SSIM:", ssim)
                    print("# Saving image...")
                    save_image(samples, 'decoded_unet_img.png')
                    save_image(images, 'decoded_unet_img_orig.png')
                    save_image(latent, 'decoded_unet_img_latent.png')

                    break



'''
TMP_3:
LPIPS: tensor(0.0336, device='cuda:0')
PSNR: tensor(17.5702, device='cuda:0')
SSIM: tensor(0.4531, device='cuda:0')
100-s
LPIPS: tensor(0.0336, device='cuda:0')
PSNR: tensor(17.5614, device='cuda:0')
SSIM: tensor(0.4515, device='cuda:0')

TMP_1:
LPIPS: tensor(0.0334, device='cuda:0')
PSNR: tensor(17.2829, device='cuda:0')
SSIM: tensor(0.4372, device='cuda:0')

TMP_2:
LPIPS: tensor(0.0327, device='cuda:0')
PSNR: tensor(17.6387, device='cuda:0')
SSIM: tensor(0.4255, device='cuda:0')

TMP_4:
LPIPS: tensor(0.0336, device='cuda:0')
PSNR: tensor(17.5641, device='cuda:0')
SSIM: tensor(0.4522, device='cuda:0')
50-s
LPIPS: tensor(0.0336, device='cuda:0')
PSNR: tensor(17.5718, device='cuda:0')
SSIM: tensor(0.4536, device='cuda:0')

'''