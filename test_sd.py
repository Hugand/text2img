from omegaconf import OmegaConf
import torch
from torch import autocast
import math
import generative_models.sgm.inference.helpers as helpers
from torchvision.utils import save_image
from generative_models.sgm.modules.diffusionmodules.sampling import EulerEDMSampler

from generative_models.sgm.util import instantiate_from_config

device = "cuda"
seed = 42
torch.manual_seed(seed)

def text_to_image(model, sampler, prompt, img_dims=256, random_seed=None, n_samples=1):
    if not random_seed == None:
        torch.manual_seed(random_seed)

    n_samples = n_samples

    H = img_dims
    W = img_dims
    C = 3
    F = 8
    n_samples = [n_samples]

    force_uc_zero_embeddings=["txt"]
    force_cond_zero_embeddings=[]
    batch2model_input = []

    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                value_dict = {
                    "txt": prompt
                }

                embs = helpers.get_unique_embedder_keys_from_conditioner(model.conditioner)
                batch, batch_uc = helpers.get_batch(
                    embs, 
                    value_dict,
                    n_samples,
                )

                # Get the conditional and unconditional text embeddings
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

                # Shape of the noise
                shape = (math.prod(n_samples), C, H, W)
                random_noise = torch.randn(shape).to(device)
                
                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )
                # Apply the reverse process
                samples = sampler(denoiser, random_noise, cond=c, uc=uc)
                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            
                return samples

def main():
    prompt = "A lion dressed as a lion"

    # model_config = OmegaConf.load("generative_models/configs/inference/sd_2_1.yaml")
    model_config = OmegaConf.load("generative_models/configs/inference/sd_2_1_768.yaml")
    model = instantiate_from_config(model_config.model)
    # sampler = instantiate_from_config(sampler_config=model_config.model.params.sampler_config)

    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    }
    # guider_config = {
    #     "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
    #     "params": {
    #         "scale": 10.0,
    #     },
    # }
    sampler = EulerEDMSampler(
        num_steps=40,
        discretization_config=discretization_config,
        # guider_config=guider_config,
        s_churn=0,
        s_tmin=0,
        s_tmax=999,
        s_noise=1,
        verbose=True,
    )
    # 84
    model = model.load_from_checkpoint(
        "checkpoints/v2-1_768-ema-pruned.safetensors",
        network_config=model_config.model.params.network_config,
        denoiser_config=model_config.model.params.denoiser_config,
        conditioner_config=model_config.model.params.conditioner_config,
        first_stage_config=model_config.model.params.first_stage_config,
        # loss_fn_config=model_config.model.params.loss_fn_config,
        # sampler_config=model_config.model.params.sampler_config,
    )
    model = model.to(device)
    # model.learning_rate = model_config.model.base_learning_rate

    samples = text_to_image(model, sampler, prompt)

    save_image(samples, 'sdxl_test_image.png')

if __name__ == "__main__":
    main()