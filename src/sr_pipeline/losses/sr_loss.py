"""L1 + 선택적 Sobel 에지 손실(A) 및 채널 라우팅(B, B′) (SPEC.md / ADR-0004).

규칙:
- 기본: L1 (화소·채널 평균)
- A 활성화 시: L1 + λ * L1(Sobel 수평·수직 성분 차이), 크기(√) 미사용
- B 또는 B′ 활성화 시: 정규화된 BT.601 Y 채널에서 두 항 계산
- 손실 계산에는 자르지 않은([0, 1] clamp 전) SR 결과를 사용

구성 스위치와 인자: A → `use_edge_loss=True`(λ는 `edge_loss_lambda`), B·Bp(B′) → `use_y_loss=True`.
B는 모델이 `sr_y_norm`(RGB로 되돌리기 전 정규화된 Y)을 주므로 그것을 쓰고, B′는 RGB SR 결과를
여기서 정규화된 Y로 바꿔 쓴다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sr_pipeline.common.color import normalize_y, rgb_to_y

# 표준 Sobel 커널(정규화하지 않음). 크기는 λ가 흡수하며 λ는 T4.2에서 고정한다.
_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))


class SobelEdgeLoss(nn.Module):
    """Sobel 수평·수직 성분 차이의 L1 손실.

    채널마다 따로 Sobel을 적용하고(복제 패딩: 영상 가장자리에 가짜 에지를 만들지 않음),
    `mean|Gx(pred) − Gx(target)| + mean|Gy(pred) − Gy(target)|`을 반환한다. 성분을 그대로 비교하고
    크기(√)는 쓰지 않는다. √는 평탄 영역에서 기울기가 발산하기 때문이다(ADR-0004).
    """

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(_SOBEL_X)
        self.register_buffer("kernels", torch.stack([kernel_x, kernel_x.T]).unsqueeze(1), persistent=False)

    def _gradients(self, image: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C, 2, H, W): 채널별 수평·수직 Sobel 성분."""
        batch, channels, height, width = image.shape
        padded = F.pad(image.reshape(batch * channels, 1, height, width), (1, 1, 1, 1), mode="replicate")
        grads = F.conv2d(padded, self.kernels.to(device=image.device, dtype=image.dtype))
        return grads.view(batch, channels, 2, height, width)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred와 target 간의 Sobel x/y 성분 차이 L1 평균을 계산합니다."""
        diff = (self._gradients(pred) - self._gradients(target)).abs()
        return diff[:, :, 0].mean() + diff[:, :, 1].mean()


class SRLoss(nn.Module):
    """스위치 A, B, Bp에 따라 채널과 항을 선택하는 통합 손실."""

    def __init__(
        self,
        use_edge_loss: bool = False,
        use_y_loss: bool = False,
        edge_loss_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        if use_edge_loss and not (math.isfinite(edge_loss_lambda) and edge_loss_lambda >= 0):
            raise ValueError(f"edge_loss_lambda는 0 이상의 유한한 수여야 합니다: {edge_loss_lambda}")
        self.use_edge_loss = use_edge_loss
        self.use_y_loss = use_y_loss
        self.edge_loss_lambda = edge_loss_lambda
        self.edge_loss = SobelEdgeLoss() if use_edge_loss else None

    def forward(
        self,
        sr_outputs: dict[str, torch.Tensor],
        hr_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """자르기 전 SR 결과와 HR RGB 간 손실을 계산합니다.

        `sr_outputs`는 모델 `forward_raw`의 반환값이다. 손실 채널은 `use_y_loss`가 꺼져 있으면 RGB,
        켜져 있으면 정규화된 Y다. 모든 항은 배치·채널·화소 평균이며, [0, 1] 자르기나 양자화를 하지 않는다.
        """
        sr_rgb = sr_outputs["sr_rgb"]
        for name, tensor in (("sr_rgb", sr_rgb), ("hr_rgb", hr_rgb)):
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(f"{name}는 (B, 3, H, W) 텐서여야 합니다: {tuple(tensor.shape)}")
            if not tensor.is_floating_point():
                raise ValueError(f"{name}는 float 텐서여야 합니다: {tensor.dtype}")
        if sr_rgb.shape != hr_rgb.shape:
            raise ValueError(f"SR 결과와 HR 영상의 shape가 다릅니다: {tuple(sr_rgb.shape)} vs {tuple(hr_rgb.shape)}")

        if "sr_y_norm" in sr_outputs and not self.use_y_loss:
            # B 모델(Y 채널 처리)에 RGB 손실을 붙인 배선 실수. B는 반드시 use_y_loss=True로 만든다.
            raise ValueError("sr_y_norm이 있는 SR 결과(스위치 B)에는 use_y_loss=True인 손실을 써야 합니다.")
        if self.use_y_loss:
            target = normalize_y(rgb_to_y(hr_rgb))
            if "sr_y_norm" in sr_outputs:  # B: 모델이 복원한 정규화된 Y
                pred = sr_outputs["sr_y_norm"]
                if pred.shape != target.shape:
                    raise ValueError(f"sr_y_norm은 {tuple(target.shape)}이어야 합니다: {tuple(pred.shape)}")
                if not pred.is_floating_point():
                    raise ValueError(f"sr_y_norm은 float 텐서여야 합니다: {pred.dtype}")
            else:  # B′: RGB SR 결과를 정규화된 Y로 바꾼다
                pred = normalize_y(rgb_to_y(sr_rgb))
        else:
            pred, target = sr_rgb, hr_rgb

        loss = F.l1_loss(pred, target, reduction="mean")
        if self.edge_loss is not None:
            loss = loss + self.edge_loss_lambda * self.edge_loss(pred, target)
        return loss
