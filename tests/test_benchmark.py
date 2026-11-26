"""추론 시간 측정 경계 테스트 (고정 입력 크기, 네트워크만 / 전체)."""

from __future__ import annotations

import math

import pytest
import torch

from sr_pipeline.evaluation.benchmark import measure_inference_time
from sr_pipeline.models.espcn import ESPCN

SMALL = {"scale": 4, "lr_height": 12, "lr_width": 16, "warmup_runs": 1, "repeat_runs": 2}


@pytest.mark.parametrize("kwargs", [{}, {"use_y_channel": True}, {"use_global_residual": True}])
def test_measure_inference_time_returns_positive_net_and_total(kwargs: dict) -> None:
    net_ms, total_ms = measure_inference_time(ESPCN(**kwargs).eval(), **SMALL)
    assert math.isfinite(net_ms) and net_ms > 0.0
    assert math.isfinite(total_ms) and total_ms > 0.0


def test_measure_inference_time_for_bicubic_has_no_network_time() -> None:
    """학습 없는 구성(model=None)은 네트워크 시간이 0이고 bicubic 확대 시간만 전체에 든다."""
    net_ms, total_ms = measure_inference_time(None, **SMALL)
    assert net_ms == 0.0
    assert math.isfinite(total_ms) and total_ms > 0.0


def test_measure_inference_time_leaves_model_state_untouched() -> None:
    model = ESPCN().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    measure_inference_time(model, **SMALL)
    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_height": 0},
        {"lr_width": -1},
        {"warmup_runs": -1},
        {"repeat_runs": 0},
        {"repeat_runs": True},
        {"scale": 0},
        {"scale": True},
    ],
)
@pytest.mark.parametrize("model", [ESPCN().eval(), None])
def test_measure_inference_time_rejects_invalid_settings(kwargs: dict, model) -> None:
    with pytest.raises(ValueError):
        measure_inference_time(model, **{**SMALL, **kwargs})


@pytest.mark.parametrize("missing", sorted(SMALL))
def test_fair_comparison_settings_have_no_defaults(missing: str) -> None:
    """측정 조건은 `fair_comparison.yaml`의 `infer_benchmark`·배율에서만 온다(함수 기본값 없음)."""
    with pytest.raises(TypeError):
        measure_inference_time(None, **{k: v for k, v in SMALL.items() if k != missing})


def test_measure_inference_time_rejects_model_without_network_stage() -> None:
    with pytest.raises(ValueError, match="_network"):
        measure_inference_time(torch.nn.Identity(), **SMALL)
