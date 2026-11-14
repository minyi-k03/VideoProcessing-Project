"""시드 하나 → 모델 초기화 난수 + 데이터 난수 테스트 (SPEC.md US32·33, ADR-0001)."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sr_pipeline.common.seed import create_data_rng, set_model_seed


@pytest.fixture(autouse=True)
def _restore_global_random_state() -> Iterator[None]:
    """set_model_seed가 바꾸는 프로세스 전역 상태를 테스트마다 되돌려 다른 테스트에 새지 않게 한다."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags


def _draw(rng: np.random.Generator) -> np.ndarray:
    """데이터 난수에서 정수 16개를 뽑는다. 16개면 서로 다른 시드의 수열이 우연히 같을 일이 없다."""
    return rng.integers(0, 2**31 - 1, size=16)


def _assert_all_distinct(items: Sequence[Any], equal: Callable[[Any, Any], bool]) -> None:
    for (i, a), (j, b) in itertools.combinations(enumerate(items), 2):
        assert not equal(a, b), f"항목 {i}와 {j}가 같다"


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
    _assert_all_distinct(draws, np.array_equal)


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


@pytest.mark.parametrize("function", [create_data_rng, set_model_seed])
def test_negative_seed_is_rejected(function: Callable[[int], object]) -> None:
    """시드는 0 이상이어야 한다. 음수가 조용히 다른 시드로 처리되면 안 된다."""
    with pytest.raises(ValueError):
        function(-1)


@pytest.mark.parametrize("function", [create_data_rng, set_model_seed])
@pytest.mark.parametrize("bad_seed", [1.5, "1", None, True])
def test_non_integer_seed_is_rejected(function: Callable[[int], object], bad_seed: object) -> None:
    """정수가 아닌 시드(실수·문자열·None·bool)는 거부한다."""
    with pytest.raises(TypeError):
        function(bad_seed)  # type: ignore[arg-type]


def test_numpy_integer_seed_is_accepted() -> None:
    """배열에서 꺼낸 numpy 정수 시드도 같은 값의 파이썬 정수와 같은 결과를 낸다."""
    np.testing.assert_array_equal(_draw(create_data_rng(np.int64(2))), _draw(create_data_rng(2)))


def _python_and_numpy_global_draws(seed: int) -> tuple[float, float]:
    set_model_seed(seed)
    return random.random(), float(np.random.random())


def test_model_seed_fixes_python_and_numpy_global_streams() -> None:
    """같은 시드면 파이썬 random·numpy 전역 난수도 같고, 시드가 다르면 다르다."""
    assert _python_and_numpy_global_draws(1) == _python_and_numpy_global_draws(1)
    _assert_all_distinct([_python_and_numpy_global_draws(seed) for seed in (0, 1, 2)], lambda a, b: a[0] == b[0])
    _assert_all_distinct([_python_and_numpy_global_draws(seed) for seed in (0, 1, 2)], lambda a, b: a[1] == b[1])


def test_model_seed_puts_cudnn_in_deterministic_mode() -> None:
    """cuDNN이 어떤 상태에서 시작하든 set_model_seed 뒤에는 결정적 모드(deterministic=True, benchmark=False)다."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_model_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_model_init_same_seed_gives_same_weights() -> None:
    """같은 시드로 만든 모델은 가중치 초기값이 정확히 같다."""
    torch.testing.assert_close(_init_weights(1), _init_weights(1), rtol=0, atol=0)


def test_model_init_different_seeds_give_different_weights() -> None:
    """시드가 다르면 가중치 초기값이 다르다."""
    weights = [_init_weights(seed) for seed in (0, 1, 2)]
    _assert_all_distinct(weights, torch.equal)
