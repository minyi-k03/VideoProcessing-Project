"""L1 + 선택적 Sobel 에지 손실(A) 및 채널 라우팅(B, B′) (SPEC.md / ADR-0004).

규칙:
- 기본: L1 (화소·채널 평균)
- A 활성화 시: L1 + λ * L1(Sobel 수평·수직 성분 차이), 크기(√) 미사용
- B 또는 B′ 활성화 시: 정규화된 BT.601 Y 채널에서 두 항 계산
- 손실 계산에는 자르지 않은([0, 1] clamp 전) SR 결과를 사용

구현 범위: 기본 L1(T2.2). 스위치 A·B·B′ 경로는 T3.5에서 구현한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Sobel 수평·수직 성분 차이의 L1 손실."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Register fixed horizontal/vertical Sobel filters.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred와 target 간의 Sobel x/y 성분 차이 L1 평균을 계산합니다."""
        raise NotImplementedError("TODO: Compute mean(|Gx(pred)-Gx(target)|) + mean(|Gy(pred)-Gy(target)|).")


class SRLoss(nn.Module):
    """스위치 A, B, Bp에 따라 채널과 항을 선택하는 통합 손실."""

    def __init__(
        self,
        use_edge_loss: bool = False,
        use_y_loss: bool = False,
        edge_loss_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_edge_loss = use_edge_loss
        self.use_y_loss = use_y_loss
        self.edge_loss_lambda = edge_loss_lambda
        if use_edge_loss or use_y_loss:
            raise NotImplementedError("TODO(T3.5): 스위치 A(Sobel 에지 손실)·B/B′(정규화된 Y 채널 손실) 경로")

    def forward(
        self,
        sr_outputs: dict[str, torch.Tensor],
        hr_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """자르기 전 SR 결과와 HR RGB 간 손실을 계산합니다.

        `sr_outputs`는 모델 `forward_raw`의 반환값이다. 기본 구성은 `sr_rgb`만 쓴다.
        배치·채널·화소 전체 평균 L1이며, [0, 1] 자르기나 양자화를 하지 않는다.
        """
        sr_rgb = sr_outputs["sr_rgb"]
        for name, tensor in (("sr_rgb", sr_rgb), ("hr_rgb", hr_rgb)):
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(f"{name}는 (B, 3, H, W) 텐서여야 합니다: {tuple(tensor.shape)}")
            if not tensor.is_floating_point():
                raise ValueError(f"{name}는 float 텐서여야 합니다: {tensor.dtype}")
        if sr_rgb.shape != hr_rgb.shape:
            raise ValueError(f"SR 결과와 HR 영상의 shape가 다릅니다: {tuple(sr_rgb.shape)} vs {tuple(hr_rgb.shape)}")
        return F.l1_loss(sr_rgb, hr_rgb, reduction="mean")
