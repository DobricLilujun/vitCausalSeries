"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class SSIMLoss_V1(nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def create_window(self, window_size, channel):
        # 生成1维高斯窗口
        def gaussian(window_size, sigma):
            sigma = torch.tensor(sigma)  # 确保 sigma 是 Tensor 类型
            gauss = torch.tensor(
                [
                    -((x - window_size // 2) ** 2) / (2 * sigma**2)
                    for x in range(window_size)
                ]
            )
            gauss = torch.exp(gauss)  # 使用 torch.exp() 对 Tensor 进行指数运算
            return gauss / gauss.sum()

        _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window @ _1d_window.T
        window = _2d_window.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, window_size, window_size)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # 计算局部均值
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # 计算局部方差
        sigma1_sq = (
            F.conv2d(img1**2, window, padding=window_size // 2, groups=channel) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2**2, window, padding=window_size // 2, groups=channel) - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        # SSIM公式中使用的常量，避免除零
        C1 = 0.01**2
        C2 = 0.03**2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.channel != channel:
            self.window = self.create_window(self.window_size, channel)
            self.channel = channel

        window = self.window.type_as(img1)
        return 1 - self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )
