import torch
from dataloaders.imagenet_loader import ImageNetLatentLoader
from torchvision.utils import save_image
device = "cuda"
torch.set_float32_matmul_precision("medium")

dataset = ImageNetLatentLoader(2, 5, val_frac=0.0001, test_frac=0.0, dims=(128, 128), random_seed=2024, crop=True)

for batch in dataset.val_dataloader():
    # torch.save(batch["jpg"], "val_img.pt")
    # torch.save(batch["ltnt"], "val_ltnt.pt")

    save_image(batch["jpg"], 'val_img.png')

    break