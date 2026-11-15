"""평가 지표 계산 모듈 (SPEC.md / ADR-0003).

주 지표: PSNR(Y), SSIM(Y) (BT.601 Y 채널)
보조 지표: PSNR(RGB), 에지 PSNR(HR Y Sobel 상위 q%), LPIPS(AlexNet, [-1, 1] RGB)
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01 * 255.0) ** 2
_SSIM_C2 = (0.03 * 255.0) ** 2


@lru_cache(maxsize=None)
def _ssim_window(device: torch.device) -> torch.Tensor:
    """BasicSR/MATLAB SSIM용 11x11 가우시안 창(sigma=1.5, float64)을 장치별로 캐시합니다."""
    with torch.inference_mode(False):
        half = _SSIM_WINDOW_SIZE // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float64)
        g1d = torch.exp(-(coords**2) / (2.0 * _SSIM_SIGMA**2))
        g1d = g1d / g1d.sum()
        return torch.outer(g1d, g1d).view(1, 1, _SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE).to(device)


def _check_image_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    labels: tuple[str, str] = ("pred", "target"),
    channels: int | None = None,
) -> None:
    """(1, C, H, W) float 영상 쌍의 shape·dtype·유한성·장치를 검증합니다."""
    for label, tensor in ((labels[0], first), (labels[1], second)):
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"{label}는 (1, C, H, W) 텐서여야 합니다. 받은 shape: {tuple(tensor.shape)}")
        if channels is not None and tensor.shape[1] != channels:
            raise ValueError(
                f"{label}는 {channels}채널 텐서여야 합니다. 받은 shape: {tuple(tensor.shape)}"
            )
        if tensor.shape[1] < 1 or tensor.shape[2] < 1 or tensor.shape[3] < 1:
            raise ValueError(f"{label}에 빈 차원이 있습니다. 받은 shape: {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"{label}는 float 텐서여야 합니다. 받은 dtype: {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label}에 NaN 또는 Inf가 포함되어 있습니다.")
    if first.shape != second.shape:
        raise ValueError(
            f"두 영상의 shape가 다릅니다: {tuple(first.shape)} vs {tuple(second.shape)}"
        )
    if first.device != second.device:
        raise ValueError(f"두 영상의 장치가 다릅니다: {first.device} vs {second.device}")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 255.0) -> float:
    """영상 1장의 PSNR을 계산합니다. 동일 영상이면 float('inf')를 반환합니다."""
    if isinstance(max_val, bool) or not isinstance(max_val, (int, float)) or not (
        math.isfinite(max_val) and max_val > 0.0
    ):
        raise ValueError(f"max_val은 0보다 큰 유한한 수여야 합니다. 받은 값: {max_val!r}")
    _check_image_pair(pred, target, labels=("pred", "target"))
    mse = (pred.double() - target.double()).pow(2).mean().item()
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((float(max_val) ** 2) / mse)


def compute_ssim_y(sr_y: torch.Tensor, hr_y: torch.Tensor) -> float:
    """가장자리 처리된 BT.601 Y 채널(16~235 범위)의 SSIM(Y)을 계산합니다."""
    _check_image_pair(sr_y, hr_y, labels=("sr_y", "hr_y"), channels=1)
    height, width = sr_y.shape[-2:]
    if height < _SSIM_WINDOW_SIZE or width < _SSIM_WINDOW_SIZE:
        raise ValueError(
            f"SSIM(Y) 계산에는 최소 {_SSIM_WINDOW_SIZE}x{_SSIM_WINDOW_SIZE} 크기가 필요합니다. "
            f"받은 크기: {height}x{width}"
        )

    img1 = sr_y.double()
    img2 = hr_y.double()
    window = _ssim_window(img1.device)

    mu1 = F.conv2d(img1, window)
    mu2 = F.conv2d(img2, window)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + _SSIM_C1) * (2.0 * sigma12 + _SSIM_C2)
    denominator = (mu1_sq + mu2_sq + _SSIM_C1) * (sigma1_sq + sigma2_sq + _SSIM_C2)
    return float((numerator / denominator).mean().item())


def compute_edge_psnr(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
    top_q_percent: float,
) -> float:
    """HR Y 채널 Sobel 크기 상위 q% 화소 영역만으로 PSNR(Y)을 계산합니다."""
    raise NotImplementedError("TODO: Compute PSNR(Y) restricted to top q% HR Sobel magnitude pixels.")


def compute_lpips(
    sr_rgb_01: torch.Tensor,
    hr_rgb_01: torch.Tensor,
    lpips_model: object | None = None,
) -> float:
    """[0, 1] RGB 영상을 [-1, 1]로 변환해 AlexNet 백본 LPIPS를 계산합니다."""
    raise NotImplementedError("TODO: Convert to [-1, 1] and evaluate AlexNet LPIPS.")
