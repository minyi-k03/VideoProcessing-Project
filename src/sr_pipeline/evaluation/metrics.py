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


_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))


def _sobel_magnitude(y: torch.Tensor) -> torch.Tensor:
    """(1, 1, H, W) Y에서 (H, W) Sobel 크기(√(gx²+gy²))를 계산합니다. 가장자리는 복제 패딩입니다."""
    kx = torch.tensor(_SOBEL_X, dtype=y.dtype, device=y.device)
    kernels = torch.stack([kx, kx.T]).unsqueeze(1)
    grad = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kernels)
    return grad.pow(2).sum(dim=1).sqrt()[0]


def _check_top_q(top_q_percent: object, name: str = "top_q_percent") -> None:
    """에지 영역 비율 q(%)가 0과 100 사이의 수인지 검증합니다(bool 제외)."""
    if (
        isinstance(top_q_percent, bool)
        or not isinstance(top_q_percent, (int, float))
        or not (0.0 < top_q_percent < 100.0)
    ):
        raise ValueError(f"{name}는 0과 100 사이의 수여야 합니다. 받은 값: {top_q_percent!r}")


def compute_edge_psnr(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
    top_q_percent: float,
) -> float:
    """HR Y 채널 Sobel 크기 상위 q% 화소 영역만으로 PSNR(Y)을 계산합니다.

    영역은 영상마다 HR Y의 Sobel 크기 상위 ceil(q% · 화소 수)개 화소다. 비율 기준이라 에지가 없는 평탄
    영상에서도 정의되며, 동률은 안정 정렬(앞 화소 우선)로 장치와 무관하게 같은 화소가 뽑힌다.
    입력은 가장자리를 이미 잘라낸 (1, 1, H, W) Y(16~235 스케일)이며 오차가 0이면 inf다.
    """
    _check_top_q(top_q_percent)
    _check_image_pair(sr_y, hr_y, labels=("sr_y", "hr_y"), channels=1)
    magnitude = _sobel_magnitude(hr_y.double()).flatten()
    count = math.ceil(magnitude.numel() * top_q_percent / 100.0)
    order = torch.sort(magnitude, descending=True, stable=True).indices[:count]
    squared = (sr_y.double() - hr_y.double())[0, 0].flatten().pow(2)
    mse = squared[order].mean().item()
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mse)


@lru_cache(maxsize=None)
def _alex_lpips(device: torch.device) -> torch.nn.Module:
    """AlexNet 백본 LPIPS(v0.1)를 장치별로 한 번만 만듭니다. 처음 만들 때 백본 가중치를 내려받습니다.

    `_ssim_window`·`common.color._constants`와 같은 이유로, 평가의 `torch.inference_mode()` 안에서
    처음 불려도 캐시에 inference 텐서가 남지 않게 inference_mode를 끄고 만든다.
    """
    import lpips

    with torch.inference_mode(False):
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def compute_lpips(
    sr_rgb_01: torch.Tensor,
    hr_rgb_01: torch.Tensor,
    lpips_model: torch.nn.Module | None = None,
) -> float:
    """[0, 1] RGB 영상을 [-1, 1]로 변환해 AlexNet 백본 LPIPS를 계산합니다.

    입력은 가장자리를 잘라낸 8비트 양자화 RGB (1, 3, H, W)다. `lpips_model`이 없으면 AlexNet LPIPS를
    만들어 쓰고(가중치 다운로드 필요), 평가가 한 번 만든 모델을 넘길 수 있다.
    """
    _check_image_pair(sr_rgb_01, hr_rgb_01, labels=("sr_rgb", "hr_rgb"), channels=3)
    model = lpips_model if lpips_model is not None else _alex_lpips(sr_rgb_01.device)
    with torch.no_grad():
        value = model(sr_rgb_01 * 2.0 - 1.0, hr_rgb_01 * 2.0 - 1.0)
    return float(value.reshape(-1)[0].item())
