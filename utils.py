import random
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from math import floor, ceil, sqrt


def seed_devices(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    
    
def layer_norm(z, eps=1e-8):
    return z / (th.sqrt(th.mean(z ** 2, dim=-1, keepdim=True)) + eps)


def plot(images, labels):
    figure = plt.figure(figsize=(5, 5))
    N = len(labels)
    n = floor(sqrt(N))
    cols, rows = n, ceil(N/n)

    for i in range(N):
        img, label = images[i], labels[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.cpu().squeeze(), cmap="gray")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def plot3ch(images, labels, class_names=None):
    figure = plt.figure(figsize=(5, 5))
    N = len(labels)
    n = floor(sqrt(N))
    cols, rows = n, ceil(N/n)

    for i in range(N):
        img, label = images[i], labels[i]
        figure.add_subplot(rows, cols, i + 1)
        if class_names:
            plt.title(class_names[label])
        else:
            plt.title(label)

        plt.axis("off")
        plt.imshow(np.transpose((img.numpy() + 1)/2, (1, 2, 0)))

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def confusion_map(actual, prediction, num_class):
    mat = []
    fig, ax = plt.subplots()

    for i in range(num_class):
        idx = th.argwhere(actual == i).flatten()
        mat.append(th.bincount(prediction[idx], minlength=num_class))

    mat = th.vstack(mat)
    ax.matshow(mat)
    for (i, j), z in np.ndenumerate(mat.numpy()):
        ax.text(j, i, z, ha='center', va='center')

    ax.set_xticks(np.arange(num_class), labels=[f"{i}" for i in range(num_class)], minor=True)
    ax.set_yticks(np.arange(num_class), labels=[f"{i}" for i in range(num_class)], minor=True)
    plt.show()
