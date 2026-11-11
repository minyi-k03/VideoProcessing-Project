"""평가 지표 계산 모듈 (SPEC.md / ADR-0003).

주 지표: PSNR(Y), SSIM(Y) (BT.601 Y 채널)
보조 지표: PSNR(RGB), 에지 PSNR(HR Y Sobel 상위 q%), LPIPS(AlexNet, [-1, 1] RGB)
"""

from __future__ import annotations

import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 255.0) -> float:
    """영상 1장의 PSNR을 계산합니다. 동일 영상이면 float('inf')를 반환합니다."""
    raise NotImplementedError("TODO: Compute PSNR with inf handling for zero MSE.")


def compute_ssim_y(sr_y: torch.Tensor, hr_y: torch.Tensor) -> float:
    """가장자리 처리된 BT.601 Y 채널(16~235 범위)의 SSIM(Y)을 계산합니다."""
    raise NotImplementedError("TODO: Implement BasicSR/MATLAB-compatible SSIM(Y).")


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
