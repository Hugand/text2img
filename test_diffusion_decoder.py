import glob
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
import torch
from dataloaders.coco17_loader import COCO17Loader
from sgm.util import instantiate_from_config

device = "cuda"
seed = 42
torch.manual_seed(seed)

model_config = OmegaConf.load("configs/models/vae_decoder_concat.yaml")
diffusion_decoder = instantiate_from_config(model_config.model)
sampler = instantiate_from_config(model_config.model.params.sampler_config)
diffusion_decoder = diffusion_decoder.load_from_checkpoint(
    "lightning_logs/version_91/checkpoints/epoch=37-step=4294.ckpt",
    network_config=model_config.model.params.network_config,
    denoiser_config=model_config.model.params.denoiser_config,
    conditioner_config=model_config.model.params.conditioner_config,
    first_stage_config=model_config.model.params.first_stage_config,
    loss_fn_config=model_config.model.params.loss_fn_config,
    sampler_config=model_config.model.params.sampler_config,
)
diffusion_decoder = diffusion_decoder.to(device)
diffusion_decoder.learning_rate = model_config.model.base_learning_rate

coco_loader = COCO17Loader(shuffle=True,
    batch_size=32, num_workers=2, dims=(128, 128),
    latents_subdir="val2017_latents_128/", test_frac=0.1)

trainer_opt = {
    "benchmark": True,
    "num_sanity_val_steps": 0,
    "accumulate_grad_batches": 1,
    "max_epochs": 300,
    "accelerator": "cuda",
    # "precision": 16,
    "val_check_interval": 0.25
}

trainer = Trainer(**trainer_opt)
out = trainer.test(
    model=diffusion_decoder,
    dataloaders=coco_loader.test_dataloader()
)

print(out)

# with torch.no_grad():
#     for dl_batch in tqdm(coco_loader.val_dataloader()):
#         samples = generate_from_decoder(diffusion_decoder, sampler, dl_batch["ltnt"].to(device), img_dims=128)
#         # samples = vae.decode(dl_batch["ltnt"].to(device))
#         # samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")
#         orig_images = torch.clamp((dl_batch["jpg"] + 1.0) / 2.0, min=0.0, max=1.0).to("cuda")

#         print(samples.shape, orig_images.shape)

#         save_image(samples, 'decoded_unet_img.png')
#         save_image(orig_images, 'decoded_unet_img_orig.png')

#         break