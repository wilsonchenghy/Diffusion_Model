import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
from torch.optim import Adam


IMG_SIZE = 64
BATCH_SIZE = 128

T = 300

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def sample_timestep_value(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())  # sample the target value from the specific timestep
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)  # out.reshape(batch_size, 1, 1, 1)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sample_timestep_value(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = sample_timestep_value(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    
    # return an image with x percent noise and x percent original image retained
    # equal to sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    # which is equivalent to sampling from the Gaussian:
    # x_t ~ N(μ = sqrt_alphas_cumprod_t * x_0, σ^2 = 1 - alphas_cumprod_t)

    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def load_transformed_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    def filter_cars(dataset):
        return [x for x in dataset if x[1] == 1]  # Label 1 = automobile

    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=data_transforms)
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transforms)

    car_data = filter_cars(train) + filter_cars(test)
    return torch.utils.data.TensorDataset(
        torch.stack([x[0] for x in car_data]),
        torch.tensor([x[1] for x in car_data])
    )

def show_tensor_image(image):
    # reverse transformation made by load_transformed_dataset()
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    plt.imshow(reverse_transforms(image))


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def forward_diffusion():
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = forward_diffusion_sample(image, t)
        show_tensor_image(img)

    plt.show()


if __name__ == "__main__":
    forward_diffusion()