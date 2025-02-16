import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from .unet import Unet


class ReconNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def pad(self, x):
        _, _, h, w = x.shape  # B, C, H, W
        hp, wp = self.net.patch_size  # Hp, Wp (Patch number)
        f1 = (
            (wp - w % wp) % wp
        ) / 2  # Padding to width needed to make it divisible by patch size
        f2 = (
            (hp - h % hp) % hp
        ) / 2  # Padding to width needed to make it divisible by patch size
        wpad = [floor(f1), ceil(f1)]
        hpad = [floor(f2), ceil(f2)]
        x = F.pad(x, wpad + hpad)

        return x, wpad, hpad

    def unpad(self, x, wpad, hpad):

        return x[..., hpad[0] : x.shape[-2] - hpad[1], wpad[0] : x.shape[-1] - wpad[1]]

    def norm(self, x):
        mean = x.view(x.shape[0], 1, 1, -1).mean(-1, keepdim=True)
        std = x.view(
            x.shape[0],
            1,
            1,
            -1,
        ).std(-1, keepdim=True)
        std = std + 1e-8
        x = (x - mean) / std
        return x, mean, std

    def unnorm(self, x, mean, std):

        return x * std + mean

    def vit_forward(self, x):
        x, wpad, hpad = self.pad(x)
        x, mean, std = self.norm(x)  # Normalize
        x = self.net(x)  # B, C, H_padded, W_padded
        x = self.unpad(x, wpad, hpad)  # B, C, H, W

        return x

    def unet_forward(self, x):
        x, mean, std = self.norm(x)
        x = self.net(x)
        x = self.unnorm(x, mean, std)

        return x

    def forward(self, x):
        if isinstance(self.net, Unet):

            return self.unet_forward(x)
        else:

            return self.vit_forward(x)
