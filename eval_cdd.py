import glob
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import torch
from dataloaders.coco17_loader import COCO17Loader
from dataloaders.imagenet_loader import ImageNetLatentLoader
from eval.reconstruction_eval import ReconstructionEval
from generate_new_decoder import generate_from_decoder
from modules.models.decoding.openai_decoder import DiffusionDecoder
from torchvision.utils import save_image
from torchvision.transforms import v2
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder

from tqdm import tqdm
import glob
import json 

device = "cuda"
seed = 42
torch.manual_seed(seed)

def ldm_transform_latent(z, extra_scale_factor=1):
    channel_means = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
    channel_stds = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

    if len(z.shape) != 4:
        raise ValueError()

    z = z * 0.18215
    channels = [z[:, i] for i in range(z.shape[1])]

    channels = [
        extra_scale_factor * (c - channel_means[i]) / channel_stds[i]
        for i, c in enumerate(channels)
    ]
    return torch.stack(channels, dim=1)

# ==== VAE ====
# model_config = OmegaConf.load("configs/models/cifar10_model.yaml")
# diffusion_decoder = load_model_from_config(model_config.first_stage_config, "checkpoints/sdxl_vae.safetensors")
# diffusion_decoder = diffusion_decoder.to("cuda")
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, device="cuda:0"
# )
# pipe.unet.cpu()
# pipe.vae.cuda()

# ==== Diffusion Decoder ====
# diffusion_decoder = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB

# model_config = OmegaConf.load("configs/models/consistency_diffusion_decoder.yaml")
# diffusion_decoder = instantiate_from_config(model_config.model)
# # Load replicated model
# cdd = DiffusionDecoder(320)
# cdd.load_state_dict(torch.load("cdd_custom.pt"))
# cdd = cdd.cuda()
# diffusion_decoder.model.diffusion_model = cdd
# sampler = instantiate_from_config(model_config.model.params.sampler_config)


model_config_original = OmegaConf.load("configs/models/consistency_diffusion_decoder.yaml")
# model_config = OmegaConf.load("configs/models/consistency_diffusion_decoder_sdxl_perceptual_loss.yaml")
diffusion_decoder = instantiate_from_config(model_config_original.model)
diffusion_decoder = diffusion_decoder.load_from_checkpoint(
    "diffusion_models_sdxl_cdd/10ske6yi/checkpoints/epoch=0-step=2950.ckpt",
    network_config=model_config_original.model.params.network_config,
    denoiser_config=model_config_original.model.params.denoiser_config,
    conditioner_config=model_config_original.model.params.conditioner_config,
    first_stage_config=model_config_original.model.params.first_stage_config,
    loss_fn_config=model_config_original.model.params.loss_fn_config,
    sampler_config=model_config_original.model.params.sampler_config,
).cuda()

loader = COCO17Loader(
    8, 10, test_frac=0.0, shuffle=False,
    dims=(256, 256), crop=True,
    latents_subdir="val2017_latents_sdxl_256/")

with torch.no_grad():
    recon_evaluator = ReconstructionEval("cuda")
    smpls = torch.Tensor([]).to("cpu")
    orig = torch.Tensor([]).to("cpu")
    metrics = {
        "lpips_vgg": [],
        "lpips_alex": [],
        "ssim": [],
        "fid": [],
        "psnr": []
    }
    i = 0
    h = 0
    for dl_batch in tqdm(loader.train_dataloader()):
        samples = generate_from_decoder(diffusion_decoder, diffusion_decoder.sampler, dl_batch["ltnt"].cuda())
        # samples = diffusion_decoder(dl_batch["ltnt"].cuda())
        # samples = pipe.vae.decode(dl_batch["ltnt"].half().cuda()).sample#.clamp(-1, 1)
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0).cuda()
        # samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        # print(samples.shape, samples.min(), samples.max())
        orig_images = torch.clamp((dl_batch["jpg"] + 1.0) / 2.0, min=0.0, max=1.0).cuda()
        # print(samples.min(), samples.max(), orig_images.min(), orig_images.max())
        # save_image(dl_batch["jpg"].cpu().float(), 'vae_out_3-1.png')
        # orig_images = dl_batch["jpg"].cuda().float()
        # print(samples.shape)
        # save_image(samples.cpu().float(), 'vae_out_s.png')
        # save_image(orig_images.cpu(), 'vae_out_o.png')
        # exit()
        # # orig_images = v2.Resize(size=samples.shape[2:])(orig_images)
        # save_image(orig_images.to("cpu").float(), 'vae_out_orig.png')

        recon_evaluator.update_fid(samples, orig_images)
        
        evals = recon_evaluator(samples.float(), orig_images.float())
        metrics["lpips_vgg"].append(evals["lpips_vgg"])
        metrics["lpips_alex"].append(evals["lpips_alex"])
        metrics["psnr"].append(evals["psnr"])
        # metrics["fid"].append(evals["fid"])
        metrics["ssim"].append(evals["ssim"])

metrics["lpips_vgg"] = torch.Tensor(metrics["lpips_vgg"]).mean().item()
metrics["lpips_alex"] = torch.Tensor(metrics["lpips_alex"]).mean().item()
metrics["ssim"] = torch.Tensor(metrics["ssim"]).mean().item()
metrics["fid"] = recon_evaluator.calc_fid().item()
metrics["psnr"] = torch.Tensor(metrics["psnr"]).mean().item()

print("VAL:", metrics)

with open("cdd_sdxl_eval_256.json", "w") as outfile: 
    json.dump(metrics, outfile)
