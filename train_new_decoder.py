from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from dataloaders.coco17_loader import COCO17Loader
from sgm.util import instantiate_from_config
import torch

from dataloaders.imagenet_loader import ImageNetLatentLoader

device = "cuda"
torch.set_float32_matmul_precision("medium")

# Phase 1 - pretraining
# dataset = CIFAR10LatentLoader(256, 2, test_frac=0.2)
# Phase 2 - Tuning for bigger imgs
# dataset = COCO17Loader(
#     batch_size=40,
#     num_workers=10,
#     test_frac=0.1,
#     dims=(128, 128),
#     latents_subdir="val2017_latents_128/"
# )
dataset = ImageNetLatentLoader(64, 10, test_frac=0.9900, dims=(128, 128), random_seed=2024)

model_config = OmegaConf.load("configs/models/ddpm_diffusion_decoder.yaml")
diffusion_decoder = instantiate_from_config(model_config.model)
# diffusion_decoder = diffusion_decoder.load_from_checkpoint(
#     "lightning_logs/version_114/checkpoints/tmp.ckpt",
#     network_config=model_config.model.params.network_config,
#     denoiser_config=model_config.model.params.denoiser_config,
#     conditioner_config=model_config.model.params.conditioner_config,
#     first_stage_config=model_config.model.params.first_stage_config,
#     loss_fn_config=model_config.model.params.loss_fn_config,
#     sampler_config=model_config.model.params.sampler_config,
# )
# diffusion_decoder = diffusion_decoder.to(device)
diffusion_decoder.learning_rate = model_config.model.base_learning_rate

trainer_opt = {
    "benchmark": True,
    "num_sanity_val_steps": 0,
    "accumulate_grad_batches": 2,
    "max_epochs": 600,
    "accelerator": "gpu",
    "precision": 16,
    "val_check_interval": 0.25
}

trainer = Trainer(**trainer_opt)
trainer.fit(
    model=diffusion_decoder,
    train_dataloaders=dataset.train_dataloader(),
    # val_dataloaders=dataset.val_dataloader()
    ckpt_path="lightning_logs/version_135/checkpoints/epoch=299-step=30300.ckpt"
)
# trainer.validate(diffusion_decoder, dataset.val_dataloader())