"""L1 + 선택적 Sobel 에지 손실(A) 및 채널 라우팅(B, B′) (SPEC.md / ADR-0004).

규칙:
- 기본: L1 (화소·채널 평균)
- A 활성화 시: L1 + λ * L1(Sobel 수평·수직 성분 차이), 크기(√) 미사용
- B 또는 B′ 활성화 시: 정규화된 BT.601 Y 채널에서 두 항 계산
- 손실 계산에는 자르지 않은([0, 1] clamp 전) 출력을 사용
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        raise NotImplementedError("TODO: Initialize L1 and optional SobelEdgeLoss.")

    def forward(
        self,
        sr_outputs: dict[str, torch.Tensor],
        hr_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """자르기 전 SR 출력과 HR RGB 간 손실을 계산합니다."""
        raise NotImplementedError("TODO: Route RGB or normalized Y channel and compute L1 (+ lambda * Sobel).")
