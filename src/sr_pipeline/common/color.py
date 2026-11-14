"""ITU-R BT.601 RGB <-> YCbCr 변환 공용 모듈 (SPEC.md / ADR-0003).

프로젝트 전체(B/B′ 학습, 진단, 평가)는 MATLAB `rgb2ycbcr` 계수와 동일한
BT.601 변환 하나만 사용합니다 (Y 범위: 16~235).

스케일 규칙:
    - RGB: [0, 1] float.
    - YCbCr: 8비트 스케일의 float (Y 16~235, Cb·Cr 16~240). BasicSR·MATLAB의 uint8 입력
      변환과 같은 스케일입니다. MATLAB에 double [0, 1]을 넣으면 결과가 255로 나눠진
      [0, 1] 스케일로 나오므로, 여기 값은 그 결과에 255를 곱한 것과 같습니다.
    - 0~255 정수로 반올림하거나 자르지 않습니다. 자르기와 양자화는
      evaluation/postprocess.py에서만 합니다.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# 8비트 스케일 BT.601 계수(MATLAB rgb2ycbcr 계수 x 255, BasicSR와 동일).
# 프로젝트에서 이 값은 이 파일에만 둡니다.
_RGB_TO_YCBCR_MATRIX = (
    (65.481, 128.553, 24.966),
    (-37.797, -74.203, 112.0),
    (112.0, -93.786, -18.214),
)
_YCBCR_OFFSET = (16.0, 128.0, 128.0)

_Y_MIN = _YCBCR_OFFSET[0]
_Y_MAX = 235.0
_Y_RANGE = _Y_MAX - _Y_MIN


@lru_cache(maxsize=None)
def _constants(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(순방향 행렬, 역행렬, 오프셋)을 device·dtype별로 한 번만 만듭니다.

    역행렬은 float64로 구한 뒤 dtype으로 내립니다. 호출마다 텐서를 새로 만들지 않아
    ADR-0003의 추론 시간(전체) 측정에 색 변환의 상수 생성 비용이 들어가지 않습니다.
    """
    forward64 = torch.tensor(_RGB_TO_YCBCR_MATRIX, dtype=torch.float64)
    forward = forward64.to(device=device, dtype=dtype)
    inverse = torch.linalg.inv(forward64).to(device=device, dtype=dtype)
    offset = torch.tensor(_YCBCR_OFFSET, device=device, dtype=dtype).view(1, 3, 1, 1)
    return forward, inverse, offset


def _check_nchw3(x: torch.Tensor, name: str) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"{name}은(는) (B, 3, H, W) 텐서여야 합니다. 받은 shape: {tuple(x.shape)}")


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)를 BT.601 YCbCr(16~235/240)로 변환합니다."""
    _check_nchw3(rgb, "rgb")
    forward, _, offset = _constants(rgb.device, rgb.dtype)
    return torch.einsum("ij,bjhw->bihw", forward, rgb) + offset


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """BT.601 YCbCr 텐서(B, 3, H, W)를 RGB([0, 1] 스케일, 자르기 전)로 변환합니다.

    순방향 행렬의 정확한 역행렬을 씁니다. MATLAB `ycbcr2rgb`의 반올림된 상수와는
    1e-4 안팎으로 다를 수 있으므로 MATLAB 역변환과의 비트 단위 일치는 주장하지 않습니다.
    """
    _check_nchw3(ycbcr, "ycbcr")
    _, inverse, offset = _constants(ycbcr.device, ycbcr.dtype)
    return torch.einsum("ij,bjhw->bihw", inverse, ycbcr - offset)


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)에서 BT.601 Y 채널(B, 1, H, W)만 추출합니다."""
    _check_nchw3(rgb, "rgb")
    forward, _, _ = _constants(rgb.device, rgb.dtype)
    weights = forward[0].view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True) + _Y_MIN


def normalize_y(y: torch.Tensor) -> torch.Tensor:
    """Y 채널(16~235 범위)을 학습 손실/입력용 [0, 1] 범위로 정규화합니다: (Y - 16) / 219."""
    return (y - _Y_MIN) / _Y_RANGE


def denormalize_y(y_norm: torch.Tensor) -> torch.Tensor:
    """정규화된 Y 채널([0, 1])을 BT.601 Y 범위(16~235)로 복원합니다."""
    return y_norm * _Y_RANGE + _Y_MIN
