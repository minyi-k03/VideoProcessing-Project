"""공통 평가 전처리 순서 모듈 (SPEC.md / ADR-0003).

모든 구성의 SR 결과는 반드시 다음 순서를 거칩니다:
1. [0, 1] 범위로 자르기 (clamp)
2. 8비트로 양자화 (uint8 변환 후 [0, 1] float 복원)
3. 가장자리 배율만큼 잘라내기 (shave, 값은 fair_comparison.yaml `shave_border`)
4. BT.601 Y 채널 변환 (`postprocess_to_y`, `common.color.rgb_to_y` 경유)
"""

from __future__ import annotations

import torch

from sr_pipeline.common.color import rgb_to_y


def quantize_and_shave(
    img: torch.Tensor,
    border: int,
) -> torch.Tensor:
    """[0, 1] 자르기 -> 8비트 양자화 -> 가장자리 잘라내기를 순차 적용합니다.

    `border`는 공정 비교 조건(`configs/fair_comparison.yaml`의 `shave_border`) 값이므로
    기본값을 두지 않고 호출하는 쪽이 명시적으로 넘깁니다. NaN 또는 Inf가 포함된 텐서는 거부합니다.
    """
    if img.ndim != 4:
        raise ValueError(f"img는 (B, C, H, W) 텐서여야 합니다. 받은 shape: {tuple(img.shape)}")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"img에 빈 배치·채널 차원이 있습니다. 받은 shape: {tuple(img.shape)}")
    if not img.is_floating_point():
        raise ValueError(f"img는 float 텐서여야 합니다. 받은 dtype: {img.dtype}")
    if not torch.isfinite(img).all():
        raise ValueError("img에 NaN 또는 Inf가 포함되어 있습니다.")
    if isinstance(border, bool) or not isinstance(border, int) or border < 0:
        raise ValueError(f"border는 0 이상의 정수여야 합니다. 받은 값: {border!r}")
    height, width = img.shape[-2:]
    if height <= 2 * border or width <= 2 * border:
        raise ValueError(
            f"영상 크기({height}x{width})가 border={border}를 잘라내기에 너무 작습니다."
        )

    clamped = img.clamp(0.0, 1.0)
    quantized = (clamped * 255.0).round().to(torch.uint8).to(img.dtype) / 255.0
    if border > 0:
        return quantized[..., border:-border, border:-border]
    return quantized


def postprocess_to_y(
    rgb: torch.Tensor,
    border: int,
) -> torch.Tensor:
    """[0, 1] 자르기 -> 8비트 양자화 -> 가장자리 잘라내기 -> BT.601 Y 변환을 적용합니다."""
    return rgb_to_y(quantize_and_shave(rgb, border=border))
