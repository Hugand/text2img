import torch
from torch import autocast
import math
import generative_models.sgm.inference.helpers as helpers

device = "cuda"


def generate_from_decoder(decoder, sampler, latent_batch, img_dims=256, random_seed=None, noise_inp=None, seed=42):
    torch.manual_seed(seed)

    if not random_seed == None:
        torch.manual_seed(random_seed)

    n_samples = len(latent_batch)

    H = latent_batch.shape[2] * 8
    W = latent_batch.shape[3] * 8
    C = 3
    F = 8
    n_samples = [n_samples]

    force_uc_zero_embeddings=["ltnt"]
    force_cond_zero_embeddings=[]
    batch2model_input = []

    with torch.no_grad():
        with autocast(device) as precision_scope:
            with decoder.ema_scope():
                value_dict = {
                    "ltnt": latent_batch
                }

                embs = helpers.get_unique_embedder_keys_from_conditioner(decoder.conditioner)
                batch, batch_uc = helpers.get_batch(
                    embs, 
                    value_dict,
                    n_samples,
                )

                # Get the conditional and unconditional text embeddings
                c, uc = decoder.conditioner.get_unconditional_conditioning(
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
                if noise_inp != None:
                    random_noise = noise_inp
                def denoiser(input, sigma, c):
                    return decoder.denoiser(
                        decoder.model, input, sigma, c, **additional_model_inputs
                    )
                # Apply the reverse process
                samples = sampler(denoiser, random_noise, cond=c, uc=uc)
                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            
                return samples
