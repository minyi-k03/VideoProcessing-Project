"""MATLAB 호환 bicubic 보간 공용 모듈 (SPEC.md / ADR-0003).

화소 중심 정렬과 축소 시 안티에일리어싱을 적용하며, GPU 텐서 연산으로 구현합니다.
사용처: 열화 모델, Cb/Cr 확대, 전역 잔차(C), Bicubic 하한, 평가.

알고리즘은 MATLAB `imresize(..., 'bicubic')`, 즉 BasicSR `matlab_functions.imresize`를 따릅니다.
    - 커널: Keys cubic(a = -0.5).
    - 좌표: 화소 중심 정렬 u = x / scale + 0.5 * (1 - 1 / scale) (x는 1부터 센 출력 좌표).
    - 축소(scale < 1)에서 안티에일리어싱을 켜면 커널 폭을 1 / scale배로 넓힙니다.
    - 가장자리는 대칭(mirror) 반사로 채우고, 출력 화소별 가중치 합을 1로 정규화합니다.
    - 출력 크기는 ceil(입력 크기 * scale)입니다.

출력은 자르지도 반올림하지도 않은 float입니다. [0, 1] 자르기와 8비트 양자화는
evaluation/postprocess.py에서만 합니다.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

_KERNEL_WIDTH = 4.0


def _cubic_kernel(x: torch.Tensor) -> torch.Tensor:
    """Keys cubic 커널(a = -0.5)."""
    absx = x.abs()
    absx2 = absx**2
    absx3 = absx**3
    near = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1)
    far = (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((absx > 1) & (absx <= 2))
    return near + far


# 가중치 행렬은 (출력 길이 x 입력 길이)라 크기별로 커질 수 있어(Urban100급 영상은 수 MB) 캐시를
# 무제한으로 두지 않는다. 학습 패치·테스트셋 크기별 항목이 들어갈 만큼만 보관한다. color._constants의
# 상수는 3x3 행렬뿐이라 크기 제한이 필요 없다.
@lru_cache(maxsize=16)
def _weights(
    in_length: int,
    scale: float,
    antialiasing: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """한 축의 리사이즈 가중치 행렬 (out_length, in_length)을 만들어 캐시합니다.

    가중치 계산은 float64로 하고 dtype으로 내립니다. 호출마다 행렬을 새로 만들지 않아
    ADR-0003의 추론 시간(전체) 측정에 가중치 생성 비용이 들어가지 않습니다.

    이 함수가 평가·추론 시간 측정의 `torch.inference_mode()` 안에서 처음 불리면, 보통의
    텐서로 만들지 않을 경우 inference 텐서가 캐시에 남아 이후 학습(B·C)의 역전파가
    "Inference tensors cannot be saved for backward"로 실패합니다. 그래서 항상
    inference_mode를 끄고 일반 텐서로 만듭니다. (common/color.py `_constants`와 같은 이유)
    """
    with torch.inference_mode(False):
        out_length = math.ceil(in_length * scale)
        shrink = scale < 1 and antialiasing
        kernel_width = _KERNEL_WIDTH / scale if shrink else _KERNEL_WIDTH

        x = torch.arange(1, out_length + 1, dtype=torch.float64)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = torch.floor(u - kernel_width / 2)
        taps = math.ceil(kernel_width) + 2
        indices = left[:, None] + torch.arange(taps, dtype=torch.float64)[None, :]
        distance = u[:, None] - indices

        if shrink:
            weights = scale * _cubic_kernel(distance * scale)
        else:
            weights = _cubic_kernel(distance)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # 1-indexed 입력 좌표를 0-indexed로 바꾸고, 범위 밖 인덱스는 대칭(mirror) 반사로 접는다.
        positions = indices.to(torch.int64) - 1
        period = 2 * in_length
        positions = positions % period
        positions = torch.where(positions >= in_length, period - 1 - positions, positions)

        matrix = torch.zeros(out_length, in_length, dtype=torch.float64)
        matrix.scatter_add_(1, positions, weights)
        return matrix.to(device=device, dtype=dtype)


def bicubic_resize(
    img: torch.Tensor,
    scale: float,
    antialiasing: bool = True,
) -> torch.Tensor:
    """MATLAB `imresize(..., 'bicubic', Antialiasing=antialiasing)` 호환 리사이즈.

    정밀도는 입력 dtype을 따릅니다. 본 실행은 FP32이며, AMP(autocast) 안에서 부르면 matmul이
    FP16으로 내려가므로 호출하는 쪽에서 autocast 밖에서 불러야 합니다.

    Args:
        img: (B, C, H, W) 형태의 float 텐서.
        scale: 배율 (예: 축소 시 0.25, 확대 시 4.0).
        antialiasing: 축소(scale < 1) 시 안티에일리어싱 적용 여부.
    """
    if img.ndim != 4:
        raise ValueError(f"img는 (B, C, H, W) 텐서여야 합니다. 받은 shape: {tuple(img.shape)}")
    if not img.is_floating_point():
        raise ValueError(f"img는 float 텐서여야 합니다. 받은 dtype: {img.dtype}")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale은 0보다 큰 유한한 수여야 합니다. 받은 값: {scale}")
    height, width = img.shape[-2:]
    if height == 0 or width == 0:
        raise ValueError(f"빈 영상은 리사이즈할 수 없습니다. 받은 shape: {tuple(img.shape)}")
    weights_h = _weights(height, float(scale), antialiasing, img.device, img.dtype)
    weights_w = _weights(width, float(scale), antialiasing, img.device, img.dtype)
    return torch.matmul(weights_h, torch.matmul(img, weights_w.transpose(0, 1)))
