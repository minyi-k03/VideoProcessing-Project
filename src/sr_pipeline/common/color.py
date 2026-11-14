"""ITU-R BT.601 RGB <-> YCbCr 변환 공용 모듈 (SPEC.md / ADR-0003).

프로젝트 전체(B/B′ 학습, 진단, 평가)는 MATLAB `rgb2ycbcr` 계수와 동일한
BT.601 변환 하나만 사용합니다 (Y 범위: 16~235).

스케일 규칙:
    - RGB: [0, 1] float.
    - YCbCr: 8비트 스케일의 float (Y 16~235, Cb·Cr 16~240). 0~255 정수로 자르거나
      반올림하지 않습니다. 자르기와 양자화는 evaluation/postprocess.py에서만 합니다.
"""

from __future__ import annotations

import torch

# MATLAB rgb2ycbcr(double, [0, 1] 입력) 계수. 프로젝트에서 이 값은 이 파일에만 둡니다.
_RGB_TO_YCBCR_MATRIX = (
    (65.481, 128.553, 24.966),
    (-37.797, -74.203, 112.0),
    (112.0, -93.786, -18.214),
)
_YCBCR_OFFSET = (16.0, 128.0, 128.0)

_Y_MIN = 16.0
_Y_RANGE = 219.0  # 235 - 16


def _check_nchw3(x: torch.Tensor, name: str) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"{name}은(는) (B, 3, H, W) 텐서여야 합니다. 받은 shape: {tuple(x.shape)}")


def _matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_RGB_TO_YCBCR_MATRIX, device=device, dtype=dtype)


def _offset(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_YCBCR_OFFSET, device=device, dtype=dtype).view(1, 3, 1, 1)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)를 BT.601 YCbCr(16~235/240)로 변환합니다."""
    _check_nchw3(rgb, "rgb")
    out = torch.einsum("ij,bjhw->bihw", _matrix(rgb.device, rgb.dtype), rgb)
    return out + _offset(rgb.device, rgb.dtype)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """BT.601 YCbCr 텐서(B, 3, H, W)를 RGB([0, 1] 스케일, 자르기 전)로 변환합니다."""
    _check_nchw3(ycbcr, "ycbcr")
    inverse = torch.linalg.inv(_matrix(ycbcr.device, torch.float64)).to(ycbcr.dtype)
    shifted = ycbcr - _offset(ycbcr.device, ycbcr.dtype)
    return torch.einsum("ij,bjhw->bihw", inverse, shifted)


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)에서 BT.601 Y 채널(B, 1, H, W)만 추출합니다."""
    _check_nchw3(rgb, "rgb")
    weights = _matrix(rgb.device, rgb.dtype)[0].view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True) + _Y_MIN


def normalize_y(y: torch.Tensor) -> torch.Tensor:
    """Y 채널(16~235 범위)을 학습 손실/입력용 [0, 1] 범위로 정규화합니다: (Y - 16) / 219."""
    return (y - _Y_MIN) / _Y_RANGE


def denormalize_y(y_norm: torch.Tensor) -> torch.Tensor:
    """정규화된 Y 채널([0, 1])을 BT.601 Y 범위(16~235)로 복원합니다."""
    return y_norm * _Y_RANGE + _Y_MIN
