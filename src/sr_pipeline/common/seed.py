"""시드 관리 및 재현성 유틸리티 (SPEC.md / ADR-0001).

시드 1개(0, 1, 2)로부터 모델 초기화 난수와 데이터 샘플러 전용 난수를 분리 파생합니다.
"""

from __future__ import annotations

import numpy as np


def set_model_seed(seed: int) -> None:
    """모델 가중치 초기화 및 학습 루프의 난수 시드를 고정합니다."""
    raise NotImplementedError("TODO: Fix torch/cuda deterministic seeds for model initialization.")


def create_data_rng(seed: int) -> np.random.Generator:
    """에폭별 패치 선택·순서·증강 전용 독립 RNG를 생성합니다."""
    raise NotImplementedError("TODO: Derive isolated data RNG from run seed.")
