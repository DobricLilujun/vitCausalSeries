"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]

import numpy as np

def sam_loss(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Spectral Angle Mapper (SAM) loss.

    Expected shapes:
    - gt, pred: [..., C] 或 [C, H, W] / [B, C, H, W] 等，只要能在最后一个维度上做通道/光谱维计算即可。
      下面实现将通道维放到最后再计算。

    Returns:
    - 标量：所有像素/样本的光谱夹角（单位：弧度）平均值，数值越小越好。
    """
    # 将通道维移到最后一维，方便广播计算
    if gt.ndim >= 3 and gt.shape[0] <= 4 and gt.shape[0] == pred.shape[0]:
        # 常见格式 [C,H,W] 或 [B,C,H,W]：把 C 轴移到最后
        axes = list(range(gt.ndim))
        gt = np.moveaxis(gt, 0 if gt.ndim == 3 else 1, -1)
        pred = np.moveaxis(pred, 0 if pred.ndim == 3 else 1, -1)
    # 现在 gt, pred 的最后一维是通道 C

    # 计算点积与范数
    dot = np.sum(gt * pred, axis=-1)                               # [...,]
    ngt = np.sqrt(np.sum(gt * gt, axis=-1))                        # [...,]
    npred = np.sqrt(np.sum(pred * pred, axis=-1))                  # [...,]

    # 余弦与裁剪（数值稳定）
    cos_sim = dot / (ngt * npred + eps)                            # [...,]
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # 夹角（弧度），并在所有位置上取均值
    angle = np.arccos(cos_sim)                                     # [...,]
    return angle.mean()



METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            args.predictions_path / tgt_file.name, "r"
        ) as recons:
            # if args.acquisition and args.acquisition != target.attrs["acquisition"]:
            #     continue

            # if args.acceleration and target.attrs["acceleration"] != args.acceleration:
            #     continue

            target = target[recons_key][()]
            recons = recons["reconstruction"][()]
            target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons = transforms.center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )
            metrics.push(target, recons)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    # parser.add_argument("--acceleration", type=int, default=None)
    # parser.add_argument(
    #     "--acquisition",
    #     choices=[
    #         "CORPD_FBK",
    #         "CORPDFS_FBK",
    #         "AXT1",
    #         "AXT1PRE",
    #         "AXT1POST",
    #         "AXT2",
    #         "AXFLAIR",
    #     ],
    #     default=None,
    #     help="If set, only volumes of the specified acquisition type are used "
    #     "for evaluation. By default, all volumes are included.",
    # )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics = evaluate(args, recons_key)
    print(metrics)
