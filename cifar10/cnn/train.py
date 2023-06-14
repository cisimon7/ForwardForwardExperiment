import sys
import torch as th
from tqdm.rich import trange
from layers import SequentialFF
from omegaconf import DictConfig
from torch.utils.data import Subset
from cifar10.dataset import get_data
from hydra import compose, initialize
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


TRAIN_NAME = "cnn_cifar10_train_03"
writer = SummaryWriter('logs/.tensorboard')


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["model=cifar10", "train=cifar10"])
        