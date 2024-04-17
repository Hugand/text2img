import glob
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pytorch_lightning as pl
import numpy as np

class COCO17Val(Dataset):
    def __init__(self,
        root_dir="./datasets/coco/",
        imgs_subdir="val2017/",
        latents_subdir="val2017_latents/",
        dims=None, #(256, 256),
        crop=False,
        test=False,
        test_frac=0.0
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dims = dims
        self.test_frac = test_frac
        self.test = test
        self.crop = crop

        self.imgs_path = root_dir + imgs_subdir
        self.latents_path = root_dir + latents_subdir
        self.file_list = glob.glob(self.imgs_path + "*")
        self.filenames = []
        for file in self.file_list:
            filename = file.split('/')[-1].split('.')[0]
            self.filenames.append(filename)

    def __getitem__(self, idx):
        if self.test:
            idx = int(idx + len(self.filenames) * (1 - self.test_frac))
        image_name = self.filenames[idx]
        image = Image.open(self.imgs_path + image_name + ".jpg")

        if self.crop:
            smallest_side = min(image.size)
            image = image.crop(((
                (image.size[0] - smallest_side) / 2,
                (image.size[1] - smallest_side) / 2,
                (image.size[0] + smallest_side) / 2,
                (image.size[1] + smallest_side) / 2,
            )))

        if self.dims != None:
            image = image.resize(self.dims)

        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)[0]
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        latent = torch.load(self.latents_path + image_name + ".pt")
        
        return { "jpg": image, "ltnt": latent }

    def __len__(self):
        if self.test:
            return int(len(self.filenames) * self.test_frac)
        else:
            return int(len(self.filenames) * (1 - self.test_frac))

class COCO17ValTmp(Dataset):
    def __init__(self,
        root_dir="./datasets/coco/",
        imgs_subdir="val2017/",
        latents_subdir="val2017_latents/",
        dims=None, #(256, 256),
        test=False,
        crop=False,
        test_frac=0.0
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dims = dims
        self.test_frac = test_frac
        self.crop = crop
        self.test = test

        self.imgs_path = root_dir + imgs_subdir
        self.latents_path = root_dir + latents_subdir
        self.file_list = glob.glob(self.imgs_path + "*")
        self.filenames = []
        for file in self.file_list:
            filename = file.split('/')[-1].split('.')[0]
            self.filenames.append(filename)

    def __getitem__(self, idx):
        if self.test:
            idx = int(idx + len(self.filenames) * (1 - self.test_frac))
        image_name = self.filenames[idx]
        image = Image.open(self.imgs_path + image_name + ".jpg")
        if self.crop:
            smallest_side = min(image.size)
            image = image.crop(((
                (image.size[0] - smallest_side) / 2,
                (image.size[1] - smallest_side) / 2,
                (image.size[0] + smallest_side) / 2,
                (image.size[1] + smallest_side) / 2,
            )))

        if self.dims != None:
            image = image.resize(self.dims)

        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)[0]
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        latent = self.latents_path + image_name + ".pt"
        
        return { "jpg": image, "ltnt": latent }

    def __len__(self):
        if self.test:
            return int(len(self.filenames) * self.test_frac)
        else:
            return int(len(self.filenames) * (1 - self.test_frac))
    

class COCO17Loader(pl.LightningDataModule):
    def __init__(self,
        batch_size,
        num_workers=0,
        shuffle=True,
        root_dir="./datasets/coco/",
        imgs_subdir="val2017/",
        latents_subdir="val2017_latents/",
        dims=None, #(256, 256),
        crop=False,
        test_frac=0.2
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = COCO17Val(
            root_dir=root_dir,
            imgs_subdir=imgs_subdir,
            latents_subdir=latents_subdir,
            crop=crop,
            dims=dims,
            test_frac=test_frac)
        self.test_dataset = COCO17Val(
            root_dir=root_dir,
            imgs_subdir=imgs_subdir,
            latents_subdir=latents_subdir,
            crop=crop,
            dims=dims,
            test=True,
            test_frac=test_frac)

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
