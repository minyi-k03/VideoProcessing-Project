"""스위치(B, C)로 제어되는 단일 ESPCN 모델 (SPEC.md / ADR-0002 / ADR-0004).

공통 구조 규칙:
- 은닉층 활성화: tanh, 마지막 층: 활성화 없음, 동일 초기화 방식
- B=False: RGB 3채널 입출력 (~37.2K params)
- B=True: 정규화된 Y 1채널만 복원 (~24.8K params) + Cb·Cr bicubic 확대 결합
- C=True: bicubic 확대 결과 + 네트워크 예측 차이 (전역 잔차)
- 바깥 인터페이스: 모든 구성에서 RGB LR (B,3,H,W) -> RGB SR (B,3,4H,4W) (자르기 전)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """개선 스위치 B·C를 독립적으로 켜고 끌 수 있는 x4 ESPCN."""

    def __init__(
        self,
        scale: int = 4,
        use_y_channel: bool = False,
        use_global_residual: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.use_y_channel = use_y_channel
        self.use_global_residual = use_global_residual
        raise NotImplementedError("TODO: Define ESPCN conv layers (tanh hidden, linear out) and common init.")

    def forward_raw(self, lr_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """RGB SR 결과(자르기 전)와 B 계열 진단용 pre-RGB 정규화 Y를 함께 반환합니다."""
        raise NotImplementedError("TODO: Implement forward pass returning {'sr_rgb': ..., 'sr_y_norm': ...}.")

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """RGB LR 텐서로부터 자르기 전 RGB SR 텐서를 반환합니다."""
        return self.forward_raw(lr_rgb)["sr_rgb"]
