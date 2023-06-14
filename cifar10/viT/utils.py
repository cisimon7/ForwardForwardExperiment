import torch as th
from torch.func import vmap
import matplotlib.pyplot as plt


def plot_patches(image, image_patches, patch_size):
    patches1 = vmap(lambda col: col.view(3, patch_size, patch_size), 0)(
        image_patches
    )
    fig = plt.figure(figsize=(15, 7.25), layout="constrained")
    sub_figs = fig.subfigures(nrows=1, ncols=2, wspace=0.07, width_ratios=[0.5, 0.5])
    
    axs0 = sub_figs[0].subplots(nrows=1, ncols=1)
    axs0.imshow(th.mul(image, 255).permute(1, 2, 0).numpy().astype("uint8"))
    axs0.axis("off")
    
    axs1 = sub_figs[1].subplots(12, 12)
    for (i, patch) in enumerate(patches1):
        axs1[i//12, i % 12].axis("off")
        axs1[i//12, i % 12].imshow(th.mul(patch, 255).permute(1, 2, 0)
                                   .numpy().astype("uint8"))
    plt.show()
    