from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import torch

device = "cuda"
torch.set_float32_matmul_precision("medium")

model_config_original = OmegaConf.load("configs/models/consistency_diffusion_decoder_ema.yaml")
# model_config = OmegaConf.load("configs/models/consistency_diffusion_decoder_sdxl_perceptual_loss.yaml")
diffusion_decoder = instantiate_from_config(model_config_original.model)
# diffusion_decoder = diffusion_decoder.load_from_checkpoint(
#     # "diffusion_models_sdxl_cdd/10ske6yi/checkpoints/epoch=0-step=2950.ckpt",
#     "diffusion_models_sdxl_cdd/y54yfs7c/checkpoints/epoch=0-step=865.ckpt",
#     network_config=model_config_original.model.params.network_config,
#     denoiser_config=model_config_original.model.params.denoiser_config,
#     conditioner_config=model_config_original.model.params.conditioner_config,
#     first_stage_config=model_config_original.model.params.first_stage_config,
#     loss_fn_config=model_config_original.model.params.loss_fn_config,
#     sampler_config=model_config_original.model.params.sampler_config,
# ).cuda()
# diffusion_decoder.learning_rate = model_config_original.model.base_learning_rate

print(diffusion_decoder.model_ema)