import glob
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
from dataloaders.coco17_loader import COCO17Loader, COCO17Val
from diffusion_decoder import DecodingUnetConcat
from generate_new_decoder import generate_from_decoder
from generative_models.sgm.inference.api import (
    SamplingParams,
    SamplingPipeline,
    SamplingSpec,
    ModelArchitecture,
)
import cv2
import generative_models.sgm.inference.helpers as helpers
import generative_models.sgm.inference.api as api
from sgm.modules.diffusionmodules.guiders import LinearPredictionGuider, VanillaCFG
from sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from sgm.util import append_dims, instantiate_from_config

from generative_models.sgm.util import disabled_train, load_model_from_config
from tqdm import tqdm

# from modules.models.decoding.decoding_unet_concat import DecodingUnetConcat

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


# # coco_loader = COCO17Loader(batch_size=1, num_workers=10, dims=(128, 128), test_frac=0)
# model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
# vae = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# vae = vae.to("cuda")
# image_name = "/home/pg51242/Desktop/text2img-gen/outputs/sdxl/base/generated_2.1.png"
# image = Image.open(image_name)
# image = image.resize((128, 128))
# image = np.array(image.convert("RGB"))
# image = image[None].transpose(0, 3, 1, 2)
# image = torch.from_numpy(image).to(dtype=torch.float32, device="cuda") / 127.5 - 1.0
# # latent = torch.load(self.latents_path + image_name + ".pt")
# print(image.shape)

# with torch.no_grad():
#     # images = torch.Tensor([])
#     # latents = torch.Tensor([])
#     latent = vae.encode(image)
    # for batch in coco_loader.train_dataloader():
    #     # print(batch["jpg"][0].shape)
    #     latent_batch = vae.encode(batch["jpg"].to("cuda"))
    #     torch.save(latent_batch[0].to("cpu"), "./datasets/coco/val2017_latents_128/" + str(batch["ltnt"][0]) + ".pt")
    #     # images_recon = torch.cat([vae.decode(latent_batch), batch.to("cpu")], dim=0)
    #     # images_dd = dd()

# del vae

model_config = OmegaConf.load("configs/models/vae_decoder_concat.yaml")
diffusion_decoder = instantiate_from_config(model_config.model)
sampler = instantiate_from_config(model_config.model.params.sampler_config)
# 84
diffusion_decoder = diffusion_decoder.load_from_checkpoint(
    "lightning_logs/version_89/checkpoints/epoch=99-step=11300.ckpt",
    network_config=model_config.model.params.network_config,
    denoiser_config=model_config.model.params.denoiser_config,
    conditioner_config=model_config.model.params.conditioner_config,
    first_stage_config=model_config.model.params.first_stage_config,
    loss_fn_config=model_config.model.params.loss_fn_config,
    sampler_config=model_config.model.params.sampler_config,
)
diffusion_decoder = diffusion_decoder.to(device)
diffusion_decoder.learning_rate = model_config.model.base_learning_rate

# lpips_list = []
# psnr_list = []
# ssim_list = []
# psnr = PeakSignalNoiseRatio().to("cuda")
i = 0

# coco = COCO17Val(latents_subdir="val2017_latents_128/", dims=(128, 128))
# a = coco[0]
# coco = COCO17Val(latents_subdir="val2017_latents/", dims=(256, 256))
# b = coco[0]

coco_loader = COCO17Loader(shuffle=False,
    batch_size=32, num_workers=2, dims=(128, 128),
    latents_subdir="val2017_latents_128/", test_frac=0.1)

# print(len(coco_loader.train_dataloader()), len(coco_loader.val_dataloader()))

with torch.no_grad():
    # samples = generate_from_decoder(diffusion_decoder, sampler, latent, img_dims=128)
    # samples = vae.decode(latent)
    # samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")
    # image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")
    # print(samples.shape, image.shape)
    # save_image(samples, 'decoded_unet_imgs_vae.png')
    # save_image(image, 'decoded_unet_img_origs_vae.png')

    for dl_batch in tqdm(coco_loader.val_dataloader()):
        samples = generate_from_decoder(diffusion_decoder, sampler, dl_batch["ltnt"].to(device), img_dims=128)
        # samples = vae.decode(dl_batch["ltnt"].to(device))
        # samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")
        orig_images = torch.clamp((dl_batch["jpg"] + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")

        print(samples.shape, orig_images.shape)

        # lpips = learned_perceptual_image_patch_similarity(samples, orig_images, net_type='squeeze')
        # psnr_score = psnr(samples, orig_images)
        # ssim = structural_similarity_index_measure(samples, orig_images)

        # # lpips_list.append(lpips)
        # # psnr_list.append(psnr)
        # # ssim_list.append(ssim)


        # print("LPIPS:", lpips)
        # print("PSNR:", psnr_score)
        # print("SSIM:", ssim)
        save_image(samples, 'decoded_unet_img.png')
        save_image(orig_images, 'decoded_unet_img_orig.png')

        break

    # final_lpips = torch.cat(lpips_list, dim=1).mean()
    # final_psnr = torch.cat(psnr_list, dim=1).mean()
    # final_ssim = torch.cat(ssim_list, dim=1).mean()

    # print("LPIPS:", final_lpips)
    # print("PSNR:", final_psnr)
    # print("SSIM:", final_ssim)

