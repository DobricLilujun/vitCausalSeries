"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


class CrossEntropyWithNaNMask(nn.Module):
    def __init__(self, num_classes, weight=None, reduction="mean"):
        super(CrossEntropyWithNaNMask, self).__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, ignore_index=num_classes
        )

    def forward(self, output, target):
        loss = self.ce_loss(output, target)
        return loss


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
        def gaussian(window_size, sigma):
            sigma = torch.tensor(sigma)
            gauss = torch.tensor(
                [
                    -((x - window_size // 2) ** 2) / (2 * sigma**2)
                    for x in range(window_size)
                ]
            )
            gauss = torch.exp(gauss)
            return gauss / gauss.sum()

        _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window @ _1d_window.T
        window = _2d_window.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, window_size, window_size)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

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

        C1 = 0.01**2
        C2 = 0.03**2

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


class MultiScaleLoss(torch.nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weight_mse=1.0, weight_sam=1.0):
        """
        Multi-Scale Loss combining MSE and SAM.

        Args:
            scales (list): Downscaling factors for multi-scale loss.
            weight_mse (float): Weight for the MSE loss component.
            weight_sam (float): Weight for the SAM loss component.
        """
        super(MultiScaleLoss, self).__init__()
        self.scales = scales
        self.weight_mse = weight_mse
        self.weight_sam = weight_sam

    def forward(self, y_true, y_pred):
        """
        Compute the combined loss.

        Args:
            y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
            y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The combined loss.
        """
        total_loss = 0.0

        for scale in self.scales:
            if scale < 1:
                y_true_scaled = F.interpolate(
                    y_true, scale_factor=scale, mode="bilinear", align_corners=False
                )
                y_pred_scaled = F.interpolate(
                    y_pred, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                y_true_scaled = y_true
                y_pred_scaled = y_pred

            # MSE Loss
            mse_loss = F.mse_loss(y_pred_scaled, y_true_scaled)

            # SAM Loss
            sam_loss = self.sam_loss(y_true_scaled, y_pred_scaled)

            # Combine losses
            total_loss += self.weight_mse * mse_loss + self.weight_sam * sam_loss

        return total_loss

    @staticmethod
    def sam_loss(y_true, y_pred):
        """
        Spectral Angle Mapper (SAM) loss.

        Args:
            y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
            y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The SAM loss.
        """
        # Flatten spatial dimensions for vectorized computation
        y_true_flat = y_true.reshape(y_true.size(0), y_true.size(1), -1)
        y_pred_flat = y_pred.reshape(y_pred.size(0), y_pred.size(1), -1)

        # Compute dot product and norms
        dot_product = torch.sum(y_true_flat * y_pred_flat, dim=1)  # (N, H*W)
        norm_true = torch.norm(y_true_flat, dim=1) + 1e-8  # Avoid division by zero
        norm_pred = torch.norm(y_pred_flat, dim=1) + 1e-8

        # Compute SAM
        sam = torch.acos(
            torch.clamp(dot_product / (norm_true * norm_pred), -1.0, 1.0)
        )  # (N, H*W)
        return torch.mean(sam)  # Mean over all pixels


import numpy as np


def spectral_angle_mapper_numpy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)

    if output.shape != target.shape:
        raise ValueError("Output and target must have the same shape.")

    B, C, H, W = output.shape

    output = output.transpose(0, 2, 3, 1).reshape(B, -1, C)
    target = target.transpose(0, 2, 3, 1).reshape(B, -1, C)

    dot_product = np.sum(output * target, axis=-1)
    norm_output = np.linalg.norm(output, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)

    cos_theta = dot_product / (norm_output * norm_target + 1e-8)

    cos_theta = np.clip(cos_theta, -1, 1)

    sam = np.arccos(cos_theta)
    sam_mean = np.mean(sam)
    return sam_mean


def peak_signal_to_noise_ratio(target, output, max_val=1.0):
    # Reshape to (Batch, Channels, H*W)
    B, C, H, W = target.shape
    target = target.reshape(B, C, -1)
    output = output.reshape(B, C, -1)

    # Compute MSE for each channel
    mse = np.mean((target - output) ** 2, axis=-1)
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-8))

    # Average PSNR over all channels and batches
    return np.mean(psnr)


def structural_similarity_index(target, output):
    B, C, H, W = target.shape
    ssim_total = 0.0

    for b in range(B):
        for c in range(C):
            # NaN mask for target and output
            target_nan_mask = np.isnan(target[b, c])
            output_nan_mask = np.isnan(output[b, c])

            # Compute mean ignoring NaNs
            target_mean = (
                np.nanmean(target[b, c]) if np.sum(~target_nan_mask) > 0 else 0.0
            )
            output_mean = (
                np.nanmean(output[b, c]) if np.sum(~output_nan_mask) > 0 else 0.0
            )

            # Replace NaNs with the computed mean
            target[b, c][target_nan_mask] = target_mean
            output[b, c][output_nan_mask] = output_mean

            # Compute data range for SSIM
            data_range = max(target[b, c].max() - target[b, c].min(), 1e-8)

            # Compute SSIM using skimage.metrics.structural_similarity
            ssim_value = ssim(
                target[b, c],
                output[b, c],
                data_range=data_range,
            )
            ssim_total += ssim_value

    # Average SSIM over batches and channels
    return ssim_total / (B * C)
