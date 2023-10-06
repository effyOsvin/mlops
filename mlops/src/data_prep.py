import os

import torch
import torchvision
from torchvision import transforms

# Path to a directory with image dataset and subfolders for training, validation and final testing
DATA_PATH = r"./mlops/data"

# Image size: even though image sizes are bigger than 64, we use this to speed up training
SIZE_H = 96
SIZE_W = 96

# Images mean and std channelwise
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# Batch size: for batch gradient descent optimization, usually selected as 2**K elements
BATCH_SIZE = 256


def data_transformer():
    return transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )


def train_data():
    transformer = data_transformer()
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "train_11k"), transform=transformer
    )
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )


def valid_data():
    transformer = data_transformer()
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "val"), transform=transformer
    )
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=0
    )


def test_data():
    transformer = data_transformer()
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "test_labeled"), transform=transformer
    )
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=0
    )
