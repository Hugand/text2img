from dataloaders.imagenet_loader import ImageNetLatentLoader
from tqdm import tqdm
import torch

dataset = ImageNetLatentLoader(
    1024, 5, val_frac=0.0, test_frac=0.0, latents_subdir="latents_train_sdxl_128/",
    dims=(128, 128), random_seed=2024, crop=True, shuffle=False)

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    # mean = torch.zeros(4).cuda()
    # std = torch.zeros(4).cuda()
    # for batch in tqdm(loader):
    #     images = batch["ltnt"].cuda()
    #     batch_size, num_channels, height, width = images.shape
    #     mean += images.mean(axis=(0, 2, 3))
    #     std += images.std(axis=(0, 2, 3))
    # mean /= 1785.0k
    # std /= 1785.0

    psum = torch.tensor([0.0, 0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(loader):
        inputs["ltnt"] = inputs["ltnt"] * 0.13025
        psum += inputs["ltnt"].sum(axis=[0, 2, 3])
        psum_sq += (inputs["ltnt"]**2).sum(axis=[0, 2, 3])

    count = len(loader.dataset) * 16 * 16

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
    return total_mean, total_std

mean, std = get_mean_std(dataset.train_dataloader())


print(mean, std)

# mean: tensor([ 0.5247, -0.0236, -0.1938,  3.1422])
# std:  tensor([7.6193, 5.8546, 7.0964, 5.4381])

# mean: tensor([ 0.0683, -0.0031, -0.0252,  0.4093])
# std:  tensor([0.9924, 0.7626, 0.9243, 0.7083])