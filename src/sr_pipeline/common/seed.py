"""시드 관리 및 재현성 유틸리티 (SPEC.md / ADR-0001).

시드 1개(0, 1, 2)로부터 모델 초기화 난수와 데이터 샘플러 전용 난수를 분리 파생합니다.
"""

from __future__ import annotations

import random

import numpy as np
import torch

# 시드에서 파생하는 난수 스트림의 번호. 번호가 다르면 같은 시드여도 서로 독립인 스트림이 된다.
_MODEL_STREAM = 0
_DATA_STREAM = 1


def set_model_seed(seed: int) -> None:
    """모델 가중치 초기화 및 학습 루프의 난수 시드를 고정합니다.

    파이썬 `random`, numpy 전역, torch(CPU·CUDA 전체)의 전역 난수를 시드에서 파생한 값으로 맞추고,
    cuDNN을 결정적 모드로 둡니다. 데이터 샘플러 난수(`create_data_rng`)는 건드리지 않습니다.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=(_MODEL_STREAM,))
    derived = int(sequence.generate_state(1)[0])  # 32비트: random·numpy 전역 시드 범위에 맞춘다.
    random.seed(derived)
    np.random.seed(derived)
    torch.manual_seed(derived)  # CPU와 모든 CUDA 장치를 함께 시드한다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_rng(seed: int) -> np.random.Generator:
    """에폭별 패치 선택·순서·증강 전용 독립 RNG를 생성합니다."""
    sequence = np.random.SeedSequence(seed, spawn_key=(_DATA_STREAM,))
    return np.random.Generator(np.random.PCG64(sequence))
