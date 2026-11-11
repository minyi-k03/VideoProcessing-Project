"""ITU-R BT.601 RGB <-> YCbCr 변환 공용 모듈 (SPEC.md / ADR-0003).

프로젝트 전체(B/B′ 학습, 진단, 평가)는 MATLAB `rgb2ycbcr` 계수와 동일한
BT.601 변환 하나만 사용합니다 (Y 범위: 16~235).
"""

from __future__ import annotations

import torch


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)를 BT.601 YCbCr(16~235/240)로 변환합니다."""
    raise NotImplementedError("TODO: Implement MATLAB-compatible BT.601 rgb_to_ycbcr on GPU/CPU tensor.")


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """BT.601 YCbCr 텐서(B, 3, H, W)를 RGB([0, 1] 스케일, 자르기 전)로 변환합니다."""
    raise NotImplementedError("TODO: Implement MATLAB-compatible BT.601 ycbcr_to_rgb on GPU/CPU tensor.")


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB 텐서(B, 3, H, W)에서 BT.601 Y 채널(B, 1, H, W)만 추출합니다."""
    raise NotImplementedError("TODO: Implement MATLAB-compatible BT.601 rgb_to_y.")


def normalize_y(y: torch.Tensor) -> torch.Tensor:
    """Y 채널(16~235 범위)을 학습 손실/입력용 [0, 1] 범위로 정규화합니다: (Y - 16) / 219."""
    raise NotImplementedError("TODO: Implement Y normalization from [16, 235] to [0, 1].")


def denormalize_y(y_norm: torch.Tensor) -> torch.Tensor:
    """정규화된 Y 채널([0, 1])을 BT.601 Y 범위(16~235)로 복원합니다."""
    raise NotImplementedError("TODO: Implement Y denormalization from [0, 1] to [16, 235].")
