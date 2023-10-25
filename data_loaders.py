import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join

warnings.filterwarnings('ignore')

class Cutout(object):

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def build_cifar(cutout=False, use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout and use_cifar10:
        aug.append(Cutout(n_holes=1, length=16))
    if cutout and not use_cifar10:
        aug.append(Cutout(n_holes=1, length=8))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root= '~/datasets', train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root= '~/datasets', train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='~/datasets', train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='~/datasets', train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset

