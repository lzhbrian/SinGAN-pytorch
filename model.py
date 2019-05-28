"""
author: lzhbrian (https://lzhbrian.me)
date: 2019.5.28
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_kernels):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(in_channels=3, out_channels=num_kernels,
                      kernel_size=(3, 3), stride=1, padding=5, bias=True),
            nn.BatchNorm2d(num_kernels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        for i in range(3):
            model.extend([
                nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                          kernel_size=(3, 3), stride=1, padding=0, bias=True),
                nn.BatchNorm2d(num_kernels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        model.extend([
            nn.Conv2d(in_channels=num_kernels, out_channels=3,
                      kernel_size=(3, 3), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])
        self.model = torch.nn.Sequential(*model)

    def forward(self, input_img, noise):
        return input_img + self.model(input_img + noise)


class Discriminator(nn.Module):
    def __init__(self, num_kernels):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(in_channels=3, out_channels=num_kernels,
                      kernel_size=(3, 3), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_kernels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        for i in range(3):
            model.extend([
                nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                          kernel_size=(3, 3), stride=1, padding=0, bias=True),
                nn.BatchNorm2d(num_kernels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        model.extend([
            nn.Conv2d(in_channels=num_kernels, out_channels=1,
                      kernel_size=(3, 3), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])
        self.model = torch.nn.Sequential(*model)

    def forward(self, input_img):
        return torch.mean(
            self.model(input_img), # Nx1xHxW
            dim=[2, 3]
        )
        # output shape: Nx1


def get_model(idx):
    num_kernels = 32 * 2 ** (idx // 4)
    print('num_kernels=', num_kernels)
    g = Generator(num_kernels)
    d = Discriminator(num_kernels)
    return g, d

