
from typing import Any
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

class ReconstructionEval:
    def __init__(self, lpips_net_type="vgg", device="cuda"):
        self.lpips_net_type = lpips_net_type
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

    def update_fid(self, samples, original):
        self.fid.update(original, real=True)
        self.fid.update(samples, real=False)

    def calc_fid(self):
        return self.fid.compute()

    def __call__(self, samples, original) -> Any:
        lpips_vgg = learned_perceptual_image_patch_similarity(
            samples, 
            original,
            net_type="vgg",
            normalize=True,
            reduction="mean"
        )
        lpips_alex = learned_perceptual_image_patch_similarity(
            samples, 
            original,
            net_type="alex",
            normalize=True,
            reduction="mean"
        )
        psnr_score = peak_signal_noise_ratio(
            samples,
            original,
            data_range=(0.0, 1.0),
        )
        ssim_score = structural_similarity_index_measure(
            samples,
            original,
            data_range=(0.0, 1.0),
        )

        return {
            "lpips_vgg": lpips_vgg,
            "lpips_alex": lpips_alex,
            "psnr": psnr_score,
            "ssim": ssim_score,
        }
    