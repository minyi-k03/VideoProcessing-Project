"""시드 하나 → 모델 초기화 난수 + 데이터 난수 테스트 (SPEC.md US32·33, ADR-0001)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sr_pipeline.common.seed import create_data_rng, set_model_seed


def _draw(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2**31 - 1, size=16)


def _init_weights(seed: int) -> torch.Tensor:
    """시드를 고정한 뒤 새 모델을 만들어 첫 합성곱 가중치를 돌려준다."""
    set_model_seed(seed)
    return nn.Conv2d(3, 8, kernel_size=5).weight.detach().clone()


def test_data_rng_same_seed_gives_same_sequence() -> None:
    """같은 시드의 데이터 난수는 항상 같은 수열을 낸다(같은 시드 → 구성 무관 같은 순서의 근거)."""
    np.testing.assert_array_equal(_draw(create_data_rng(0)), _draw(create_data_rng(0)))


def test_data_rng_different_seeds_give_different_sequences() -> None:
    """시드 0·1·2의 데이터 난수 수열은 서로 모두 다르다."""
    draws = [_draw(create_data_rng(seed)) for seed in (0, 1, 2)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.array_equal(draws[i], draws[j])


def test_data_sequence_is_independent_of_model_side_randomness() -> None:
    """모델 초기화·전역 난수 소비량이 달라도 같은 시드의 데이터 수열은 변하지 않는다(구성 무관 같은 순서)."""
    reference = _draw(create_data_rng(2))

    rng = create_data_rng(2)  # 모델 쪽 난수를 쓰기 전에 만들어 둔 RNG도 영향받지 않아야 한다.
    set_model_seed(2)
    nn.Conv2d(3, 64, kernel_size=5)  # 구성마다 다른 모델 → 다른 양의 전역 난수 소비
    torch.randn(1000)
    np.random.random(1000)
    after_model_use = _draw(rng)
    fresh_after_model_use = _draw(create_data_rng(2))

    np.testing.assert_array_equal(after_model_use, reference)
    np.testing.assert_array_equal(fresh_after_model_use, reference)


def test_data_rng_does_not_touch_model_side_global_state() -> None:
    """데이터 난수를 만들고 쓰는 일이 모델 쪽 전역 난수 수열을 바꾸지 않는다."""
    set_model_seed(0)
    expected = torch.randn(8)

    set_model_seed(0)
    _draw(create_data_rng(0))
    actual = torch.randn(8)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_each_data_rng_call_returns_independent_generator() -> None:
    """같은 시드로 만든 두 RNG는 서로 상태를 공유하지 않는다(한쪽을 소비해도 다른 쪽은 처음부터)."""
    first = create_data_rng(1)
    second = create_data_rng(1)
    _draw(first)
    np.testing.assert_array_equal(_draw(second), _draw(create_data_rng(1)))


def test_negative_seed_is_rejected() -> None:
    """시드는 0 이상 정수여야 한다. 음수가 조용히 다른 시드로 처리되면 안 된다."""
    with pytest.raises(ValueError):
        create_data_rng(-1)
    with pytest.raises(ValueError):
        set_model_seed(-1)


def test_model_init_same_seed_gives_same_weights() -> None:
    """같은 시드로 만든 모델은 가중치 초기값이 정확히 같다."""
    torch.testing.assert_close(_init_weights(1), _init_weights(1), rtol=0, atol=0)


def test_model_init_different_seeds_give_different_weights() -> None:
    """시드가 다르면 가중치 초기값이 다르다."""
    weights = [_init_weights(seed) for seed in (0, 1, 2)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(weights[i], weights[j])
