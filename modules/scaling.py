from abc import ABC, abstractmethod
from typing import Tuple
import math
import torch

# class DenoiserScaling(ABC):
#     @abstractmethod
#     def __call__(
#         self, sigma: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         pass



# alphas = 1.0 - betas
# alphas_cumprod = torch.cumprod(alphas, dim=0)
# self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
# sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
# sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
# self.c_skip = (
#     sqrt_recip_alphas_cumprod
#     * sigma_data**2
#     / (sigmas**2 + sigma_data**2)
# )
# self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
# self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

class CDDScaling:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_cumprod = 1.0 / (sigma**2 + 1.0)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alpha_cumprod)
        c_skip = sqrt_recip_alphas_cumprod * self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = sqrt_recip_alphas_cumprod / (sigma**2 + self.sigma_data**2) ** 0.5
        # c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in#, c_noise

