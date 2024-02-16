import torch
from torch.utils.data import DataLoader, Dataset
import math
import pytorch_lightning as pl

class CIFAR10LatentsDataDictWrapper(Dataset):
    def __init__(self, images, latents):
        super().__init__()
        self.images = images
        self.latents = latents

    def __getitem__(self, i):
        return { "jpg": self.images[i], "ltnt": self.latents[i] }

    def __len__(self):
        return len(self.images)
    

class CIFAR10LatentLoader(pl.LightningDataModule):
    def __init__(self,
        batch_size,
        num_workers=0,
        shuffle=True,
        root_dir="./datasets/cifar10-latents/",
        images_file="cifar10-imgs.pt",
        latents_file="cifar10-latents.pt",
        test_frac=0.2
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        images = torch.load(root_dir + images_file)
        latents = torch.load(root_dir + latents_file)
        test_size = math.floor(images.shape[0] * test_frac)

        self.train_dataset = CIFAR10LatentsDataDictWrapper(images[:test_size], latents[:test_size])
        self.test_dataset = CIFAR10LatentsDataDictWrapper(images[test_size:], latents[test_size:])

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
