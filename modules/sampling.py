"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from generative_models.sgm.util import instantiate_from_config,default

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        # x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class CDDSampler(SingleStepDiffusionSampler):
    def __init__(
        self, n_schedule_steps, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_schedule_steps = n_schedule_steps
        self.schedule = torch.flip(torch.arange(0, 1.0, 1/n_schedule_steps), [0]) + 1/n_schedule_steps
        

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L895    """
        res = arr[timesteps].float()
        dims_to_append = len(broadcast_shape) - len(res.shape)
        return res[(...,) + (None,) * dims_to_append]

    def sampler_step(self, step, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        noise = torch.randn_like(x)
        timestep = torch.tensor([step] * x.shape[0])
        alphas_cumprod = self.discretization.to_torch(self.discretization.alphas_cumprod).to(self.device)
        sqrt_alpha_cumprod = self._extract_into_tensor(torch.sqrt(alphas_cumprod), timestep, x.shape)
        sqrt_one_minus_alpha_cumprod = self._extract_into_tensor(torch.sqrt(1 - alphas_cumprod), timestep, x.shape)
        x = (
            sqrt_alpha_cumprod * x 
            + sqrt_one_minus_alpha_cumprod * noise
        )

        x = self.denoise(x, denoiser, sigma, cond, uc)

        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        schedule_timesteps = [int((self.num_steps - 1) * s) for s in self.schedule]

        for i in schedule_timesteps:
            x = self.sampler_step(
                i,
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x
