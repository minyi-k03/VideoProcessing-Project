"""스위치(B, C)로 제어되는 단일 ESPCN 모델 (SPEC.md / ADR-0002 / ADR-0004).

공통 구조 규칙:
- 은닉층 활성화: tanh, 마지막 층: 활성화 없음, 동일 초기화 방식
- B=False: RGB 3채널 입출력 (~37.2K params)
- B=True: 정규화된 Y 1채널만 복원 (~24.8K params) + Cb·Cr bicubic 확대 결합
- C=True: bicubic 확대 결과 + 네트워크 예측 차이 (전역 잔차)
- 바깥 인터페이스: 모든 구성에서 RGB LR (B,3,H,W) -> RGB SR (B,3,4H,4W) (자르기 전)

층 구성 (Shi et al., 2016):
    conv 5x5, c -> 64, tanh
    conv 3x3, 64 -> 32, tanh
    conv 3x3, 32 -> c * scale^2  (활성화 없음)
    pixel shuffle (sub-pixel), scale
c = 3 (RGB) 또는 1 (B: 정규화된 Y). x4에서 RGB 37,200개, Y 24,752개 파라미터.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

SCALE = 4
HIDDEN_CHANNELS = (64, 32)
KERNEL_SIZES = (5, 3, 3)
LAST_LAYER_INIT_STD = 1e-3


class ESPCN(nn.Module):
    """개선 스위치 B·C를 독립적으로 켜고 끌 수 있는 x4 ESPCN."""

    def __init__(
        self,
        scale: int = 4,
        use_y_channel: bool = False,
        use_global_residual: bool = False,
    ) -> None:
        super().__init__()
        if scale != SCALE:
            raise ValueError(f"이 프로젝트의 배율은 x{SCALE} 고정입니다: {scale}")
        self.scale = scale
        self.use_y_channel = use_y_channel
        self.use_global_residual = use_global_residual

        channels = 1 if use_y_channel else 3
        c1, c2 = HIDDEN_CHANNELS
        k1, k2, k3 = KERNEL_SIZES
        self.feature = nn.Sequential(
            nn.Conv2d(channels, c1, k1, padding=k1 // 2),
            nn.Tanh(),
            nn.Conv2d(c1, c2, k2, padding=k2 // 2),
            nn.Tanh(),
        )
        self.last = nn.Conv2d(c2, channels * scale**2, k3, padding=k3 // 2)
        self.upsample = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self) -> None:
        """모든 구성이 공유하는 초기화 방식.

        은닉층: N(0, sqrt(2 / (out_channels * k * k))), 마지막 층: N(0, 1e-3), 편향 0.
        난수는 모델 생성 전에 `common.seed.set_model_seed`로 고정한다.
        """
        for module in self.feature:
            if isinstance(module, nn.Conv2d):
                fan_out = module.out_channels * module.weight[0][0].numel()
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.last.weight, mean=0.0, std=LAST_LAYER_INIT_STD)
        nn.init.zeros_(self.last.bias)

    def _network(self, x: torch.Tensor) -> torch.Tensor:
        """LR 공간 특징 추출 → sub-pixel 확대. 입력·출력 채널 수는 같다(RGB 3 또는 Y 1)."""
        return self.upsample(self.last(self.feature(x)))

    def forward_raw(self, lr_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """RGB SR 결과(자르기 전)와 B 계열 진단용 pre-RGB 정규화 Y를 함께 반환합니다.

        `sr_y_norm`은 B가 켜진 구성에서만 들어간다(T3.4).
        """
        if lr_rgb.ndim != 4 or lr_rgb.shape[1] != 3:
            raise ValueError(f"lr_rgb는 (B, 3, H, W) 텐서여야 합니다: {tuple(lr_rgb.shape)}")
        if not lr_rgb.is_floating_point():
            raise ValueError(f"lr_rgb는 float 텐서여야 합니다: {lr_rgb.dtype}")
        if self.use_y_channel or self.use_global_residual:
            raise NotImplementedError("TODO(T3.4): 스위치 B(Y 채널)·C(bicubic 확대 + 예측 차이) 경로")
        return {"sr_rgb": self._network(lr_rgb)}

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """RGB LR 텐서로부터 자르기 전 RGB SR 텐서를 반환합니다."""
        return self.forward_raw(lr_rgb)["sr_rgb"]
