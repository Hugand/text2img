from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from dataloaders.coco17_loader import COCO17Loader
from sgm.util import instantiate_from_config
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorch_lightning as L
from dataloaders.imagenet_loader import ImageNetLatentLoader
from modules.models.decoding.openai_decoder import DiffusionDecoder

device = "cuda"
torch.set_float32_matmul_precision("medium")

# dataset = ImageNetLatentLoader(64, 10, test_frac=0.99, dims=(128, 128), random_seed=2024)

# model_config = OmegaConf.load("configs/models/ddpm_diffusion_decoder.yaml")
# diffusion_decoder = instantiate_from_config(model_config.model)
# diffusion_decoder = diffusion_decoder.load_from_checkpoint(
#     "lightning_logs/version_114/checkpoints/tmp.ckpt",
#     network_config=model_config.model.params.network_config,
#     denoiser_config=model_config.model.params.denoiser_config,
#     conditioner_config=model_config.model.params.conditioner_config,
#     first_stage_config=model_config.model.params.first_stage_config,
#     loss_fn_config=model_config.model.params.loss_fn_config,
#     sampler_config=model_config.model.params.sampler_config,
# diffusion_decoder = diffusion_decoder.to(device)
import time

# while True:
#     if torch.cuda.mem_get_info(device="cuda:0")[0] >= 15000000000:
#         breakll
#     print("Scanning...")
#     time.sleep(10)


# model_config_original = OmegaConf.load("configs/models/consistency_diffusion_decoder_ema.yaml")
# # model_config = OmegaConf.load("configs/models/consistency_diffusion_decoder_sdxl_perceptual_loss.yaml")
# diffusion_decoder = instantiate_from_config(model_config_original.model)
# diffusion_decoder = diffusion_decoder.load_from_checkpoint(
#     # "diffusion_models_sdxl_cdd/10ske6yi/checkpoints/epoch=0-step=2950.ckpt",
#     "diffusion_models_sdxl_cdd/femtjxre/checkpoints/epoch=0-step=75.ckpt",
#     network_config=model_config_original.model.params.network_config,
#     denoiser_config=model_config_original.model.params.denoiser_config,
#     conditioner_config=model_config_original.model.params.conditioner_config,
#     first_stage_config=model_config_original.model.params.first_stage_config,
#     loss_fn_config=model_config_original.model.params.loss_fn_config,
#     sampler_config=model_config_original.model.params.sampler_config,
# ).cuda()
# diffusion_decoder.learning_rate = model_config_original.model.base_learning_rate

# loss_fn = instantiate_from_config(model_config.model.params.loss_fn_config)
# diffusion_decoder.loss_fn = loss_fn


model_config = OmegaConf.load("configs/models/consistency_diffusion_decoder_ema.yaml")
diffusion_decoder = instantiate_from_config(model_config.model)
# Load replicated model
# diffusion_decoder.model.diffusion_model = DiffusionDecoder(320)
# diffusion_decoder.model.diffusion_model.load_state_dict(torch.load("cdd_custom.pt"))
# diffusion_decoder.model.diffusion_model = diffusion_decoder.model.diffusion_model.cuda()
diffusion_decoder.learning_rate = model_config.model.base_learning_rate


# Load replicated model
# sampler = instantiate_from_config(model_config.model.params.sampler_config)


batch_size = 2
accumulated_grad = 80
effective_batch_size = batch_size * accumulated_grad

# Logger
wandb_logger = WandbLogger(
    project='diffusion_models_sdxl_cdd',
    name='cdd-sdxl-finetune-128-ema',
    id='femtjxre',
    # log_model='all',
    resume='allow'
)
wandb_logger.experiment.config["batch_size"] = effective_batch_size
wandb_logger.experiment.config["learning_rate"] = diffusion_decoder.learning_rate
# checkpoint_callback = L.callbacks.ModelCheckpoint(
#     monitor='val/loss',  # Replace with your validation metric
#     mode='min',          # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
#     save_top_k=2,        # Save top k checkpoints based on the monitored metric
#     save_last=True,      # Save the last checkpoint at the end of training
# )

dataset = ImageNetLatentLoader(
    batch_size, 5, val_frac=0.00035, test_frac=0.0, latents_subdir="latents_train_sdxl_128/",
    dims=(128, 128), random_seed=2024, crop=True, shuffle=False)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="diffusion_models_sdxl_cdd",
    
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": diffusion_decoder.learning_rate,
#         "architecture": "CDD-SDXL",
#         "epochs": 1,
#         "batch_size": effective_batch_size
#     }
# )

# diffusion_decoder.set_wandb(wandb)

trainer_opt = {
    "benchmark": True,
    "num_sanity_val_steps": 0,
    "accumulate_grad_batches": accumulated_grad,
    "max_epochs": 2,
    "accelerator": "gpu",
    "precision": "16-mixed",
    "val_check_interval": accumulated_grad * 5, # 400
    "logger": wandb_logger,
    "log_every_n_steps": 1,
    # "callbacks": [checkpoint_callback]
}
# /home/pg51242/Desktop/text2img-gen/lightning_logs/version_136/
print("Train size:" ,len(dataset.train_dataloader().dataset))
print("Val size:", len(dataset.val_dataloader().dataset))
print("Total:", len(dataset.train_dataloader().dataset) + len(dataset.val_dataloader().dataset))
trainer = Trainer(**trainer_opt)

trainer.fit(
    model=diffusion_decoder,
    train_dataloaders=dataset.train_dataloader(),
    val_dataloaders=dataset.val_dataloader(),
    ckpt_path="diffusion_models_sdxl_cdd/femtjxre/checkpoints/epoch=0-step=985.ckpt"
)
# trainer.validate(diffusion_decoder, dataset.val_dataloader()
