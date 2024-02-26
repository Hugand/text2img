import torch
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder, save_image, load_image

decoder_consistency = ConsistencyDecoder(device="cpu") # Model size: 2.49 GB


print(decoder_consistency)