"""MATLAB 호환 bicubic 보간 공용 모듈 (SPEC.md / ADR-0003).

화소 중심 정렬과 축소 시 안티에일리어싱을 적용하며, GPU 텐서 연산으로 구현합니다.
사용처: 열화 모델, Cb/Cr 확대, 전역 잔차(C), Bicubic 하한, 평가.
"""

from __future__ import annotations

import torch


def bicubic_resize(
    img: torch.Tensor,
    scale: float,
    antialiasing: bool = True,
) -> torch.Tensor:
    """MATLAB `imresize(..., 'bicubic', Antialiasing=antialiasing)` 호환 리사이즈.

    Args:
        img: (B, C, H, W) 형태의 float 텐서.
        scale: 배율 (예: 축소 시 0.25, 확대 시 4.0).
        antialiasing: 축소(scale < 1) 시 안티에일리어싱 적용 여부.
    """
    raise NotImplementedError("TODO: Implement MATLAB-compatible GPU tensor bicubic resize.")
