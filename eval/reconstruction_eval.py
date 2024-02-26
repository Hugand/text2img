
from typing import Any
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio

class ReconstructionEval:
    def __init__(self, lpips_net_type="vgg", device="cuda"):
        self.lpips_net_type = lpips_net_type
        self.psnr = PeakSignalNoiseRatio().to(device)


    def __call__(self, samples, original) -> Any:
        lpips = learned_perceptual_image_patch_similarity(samples, original, net_type='squeeze')
        psnr_score = self.psnr(samples, original)
        ssim = structural_similarity_index_measure(samples, original)

        return {
            "lpips": lpips,
            "psnr": psnr_score,
            "ssim": ssim
        }