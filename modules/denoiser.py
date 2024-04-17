from typing import Dict, Union

import torch
import torch.nn as nn

from generative_models.sgm.util import append_dims, instantiate_from_config


class CDDDenoiser(nn.Module):
    def __init__(self, scaling_config: Dict, discretization_config: Dict):
        super().__init__()

        self.discretization = instantiate_from_config(discretization_config)
        self.sigmas = self.discretization(self.discretization.num_timesteps-1)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise
    
    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = self.sigmas.to("cuda") - sigma[0][0][0].to("cuda")
        return dists.abs().argmin(dim=0).item()

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        # sqrt_recip_alphas_cumprod: torch.Tensor,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in = self.scaling(sigma) #, sqrt_recip_alphas_cumprod)
        # c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        timestep = torch.Tensor([self.sigma_to_idx(sigma)] * input.shape[0]).cuda()
        out = network(c_in * input, timestep, cond,  **additional_model_inputs)
        out = (c_out * out + c_skip * input).clamp(-1, 1)
        return out

