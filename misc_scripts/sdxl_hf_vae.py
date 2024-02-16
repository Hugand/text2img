import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

device = "cuda"


# load both base & refiner
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae",
    torch_dtype=torch.float16)
vae.to(device)

# Open image
image = Image.open("sdxl_hf.png")
#image = np.asarray(image)

w, h = image.size
print(f"loaded input image of size ({w}, {h})")
width, height = map(
    lambda x: x - x % 64, (w, h)
)  # resize to integer multiple of 64
image = image.resize((width, height))
image = np.array(image.convert("RGB"))
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image).to(dtype=torch.float16) / 127.5 - 1.0
image = image.to(device)


with torch.no_grad():
    print("# Encoding...")
    latent = vae.encode(image)

    print(latent.latent_dist.sample())

    



    print("# Decoding...")
    out = vae.decode(latent)

    print(out)

save_image(out, "decoded.png")
