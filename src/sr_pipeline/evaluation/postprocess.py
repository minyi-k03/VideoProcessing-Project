"""공통 평가 전처리 순서 모듈 (SPEC.md / ADR-0003).

모든 구성의 SR 결과는 반드시 다음 순서를 거칩니다:
1. [0, 1] 범위로 자르기 (clamp)
2. 8비트로 양자화 (uint8 변환 후 float 복원 또는 [0, 255] 표현)
3. 가장자리 4화소(배율만큼) 잘라내기 (shave)
"""

from __future__ import annotations

import torch


def quantize_and_shave(
    img: torch.Tensor,
    border: int = 4,
) -> torch.Tensor:
    """[0, 1] 자르기 -> 8비트 양자화 -> 가장자리 잘라내기를 순차 적용합니다."""
    raise NotImplementedError("TODO: Clamp [0, 1], quantize to 8-bit, and shave border pixels.")
