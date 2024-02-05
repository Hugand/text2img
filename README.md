# Text-to-Image Generation with Diffusion Models

## SDXL datasets
### Training
256x256, augmented with random crops, flips and rotations

- ImageNet: 1.8 million images, 1000 object categories
- OpenImages: 1.2 million images, 600 oject categories

### Evaluation

- COCO: 330000 images, 80 objects
- ImageNet
- LSUN

Metrics:
- FID
- IS
- Learned Perceptual Image Patch Similarity (LPIPS)

## SDXL Future work:

- Single stage: Currently, we generate the best samples from SDXL using a two-stage approach
with an additional refinement model. This results in having to load two large models into
memory, hampering accessibility and sampling speed. Future work should investigate ways
to provide a single stage of equal or better quality.

- Text synthesis: While the scale and the larger text encoder (OpenCLIP ViT-bigG)
help to improve the text rendering capabilities over previous versions of Stable Diffusion,
incorporating byte-level tokenizers or simply scaling the model to larger sizes may further improve text synthesis.

- Architecture: During the exploration stage of this work, we briefly experimented with
transformer-based architectures such as UViT and DiT, but found no immediate
benefit. We remain, however, optimistic that a careful hyperparameter study will eventually
enable scaling to much larger transformer-dominated architectures.

- Distillation: While our improvements over the original Stable Diffusion model are significant,
they come at the price of increased inference cost (both in VRAM and sampling speed).
Future work will thus focus on decreasing the compute needed for inference, and increased
sampling speed, for example through guidance-, knowledge-  and progressive
distillation. [FIXED WITH SDXL-TURBO]

- Our model is trained in the discrete-time formulation of DDPM, and requires offset-noise for aesthetically pleasing results. The EDM-framework of Karras et al. is a promising candidate for future model training, as its formulation in continuous time allows for increased sampling flexibility and does not require noise-schedule corrections.

## SDXL Limitations:

- Struggles to generate intricate structures, like hands.
- Still doesn't achieve perfect photorealistic images.
- Relying on large-scale datasets might introduce some social and racial biases.
- Exhibits the concept bleeding phenomenon, which refers to having difficulties generating images with multiple objects and concepts, merging or overlaping distinct visual elements.
- Struggles rendering text in images. Might be fixed character-level text encoders.

## SDXL Improvements (attempt) List:

- VAE similar to DALL-E 3: Replace the decoder of the VAE with a DDPM and apply Adversarial Diffusion Distillatio instead of Consistency distillation. Or actually try both methods. DALL-E 3 saw improvements in fine details of the images.

- Image captioner: To try handling longer and highly descriptive text prompts, train a image captioner to generate prompts from images and build a dataset to train the text2img model, similar to DALL-E 3.

- Transform a two-stage model into a one-stage model: Can the U-Net be conditioned to act as a base model or as a refiner and use a single model to perform a two-stage inference (base + refinement)?

- Experiment with character-level text encoders to improve rendered text in the images.