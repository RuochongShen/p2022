import random
import imageio
import numpy as np
from argparse import ArgumentParser

import torchvision.utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure

from medical.ctbasic import *
from models import *
from metrics import *

# Import of custom models
from models.diffusion import MyDDPM
from models.UNet import MyUNet
from datasets import *
from losses import *

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def show_first_batch(loader):
    for i, (mt, gt) in (enumerate(loader)):
        print(mt.size(), gt.size())
        show_images(mt, "Images with metal in the first batch")
        show_images(gt, "Images without metal in the first batch")
        break


def show_forward(ddpm, loader, device):
    # Showing the forward process
    for i, (mt, gt) in enumerate(loader):
        imgs = gt

        show_images(imgs, "Original images without metal")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break


def generate_new_images(ddpm, condx, im_gt=None, n_samples=1, device=None, frames_per_gif=100, folder_name="sampling", c=1, h=256, w=256, threshold_m=1):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    pics_idxs = np.linspace(0, ddpm.n_steps, 10).astype(np.uint)
    frames = []
    pic_id = 0

    if not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor, condx)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x_metal = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            eps = torch.randn(n_samples, c, h, w).to(device)
            x_nometal = alpha_t_bar.sqrt() * condx + (1 - alpha_t_bar) * eps

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x_metal = x_metal + sigma_t * z

            x = x_metal * (condx > threshold_m) + x_nometal * (condx <= threshold_m)

            if idx in pics_idxs or t == 0:
                MAX_pic = 1 if im_gt is None else torch.max(im_gt)
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= MAX_pic / torch.max(normalized[i])
                # normalized = normalized.to(torch.uint8)
                torchvision.utils.save_image(torch.mean(normalized[:, 0, :, :], 0),
                                             folder_name + "/" + str(pic_id) + ".png")
                pic_id += 1

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(folder_name + "/" + "sampling.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])

    # Storing raw/ground truth if exists
    torchvision.utils.save_image(condx, folder_name + "/" + "raw.png")
    if im_gt is not None:
        torchvision.utils.save_image(im_gt, folder_name + "/" + "gt.png")

    x = torch.mean(x, 0)

    scipy.io.savemat(folder_name + "/" + "imData.mat",
                     {
                         "imRaw": condx.cpu().numpy(),
                         "imRef": im_gt.cpu().numpy(),
                         "imDDPM": x.cpu().numpy(),
                         # "metalBW": chosen_arr.cpu().numpy()
                     })

    return x


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    losses = np.array([])

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, (mt, gt) in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = gt.to(device)
            n = len(x0)
            mt = mt.to(device)
            myweight = mt - torch.min(mt)
            myweight = myweight / torch.max(myweight)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1), condx=mt)

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta*myweight, eta*myweight)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        #if display:
        #    show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.8f}"
        losses = np.append(losses, epoch_loss)
        torch.save(losses, "training_loss.pt")

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

    # save loss
    torch.save(losses, "training_loss.pt")
    plt.plot(np.arange(n_epochs)+1, losses)