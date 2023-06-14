import torch as th
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from omegaconf import DictConfig
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def get_cifar10(size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.02 * 2 * th.pi, interpolation=transforms.InterpolationMode.NEAREST)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    ])
    train_set = CIFAR10(
        root='../datasets',
        train=True,                         
        transform=transform,
        download=True,            
    )
    test_set = CIFAR10(
        root='../datasets', 
        train=False, 
        transform=transform_test
    )
    return train_set, test_set


def get_data(size, batch_size=100, num_workers=0):
    train_set, test_set = get_cifar10(size)
    train_set, test_set = CIFAR10Data(train_set), CIFAR10Data(test_set)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    return train_set, test_set, train_loader, test_loader


class CIFAR10Data(Dataset):
    def __init__(self, cifar10: Dataset):
        self.cifar10 = cifar10
        self.neu_label = th.ones(10) / 10
        self.classes = th.arange(0, 10)
        self.class_names = (
            'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        pos_label = th.tensor(label)
        neg_label = self.classes[self.classes != label][th.randint(0, 9, (1,))].squeeze()
        return image, (
            th.nn.functional.one_hot(pos_label, num_classes=10),
            th.nn.functional.one_hot(neg_label, num_classes=10),
            self.neu_label
        )

    def __len__(self):
        return len(self.cifar10)

    