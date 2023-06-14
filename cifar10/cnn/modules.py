import torch as th


def cnn_block(inp_channels, out_channels):
    return th.nn.Sequential(
        th.nn.Conv2d(inp_channels, out_channels, 3, 1, 1),
        th.nn.ReLU(True),
        th.nn.MaxPool2d(kernel_size=2, stride=2),
        th.nn.BatchNorm2d(out_channels, eps=1e-4)
    )
