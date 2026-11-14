"""베이스라인 ESPCN 바깥 동작 테스트 (SPEC.md US21·75, ADR-0002, Task T2.1).

합성 데이터만 쓰므로 로컬 CPU에서 돈다. 스위치 B·C 동작은 `test_model_and_loss.py`(T3.4)에서 검증한다.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sr_pipeline.evaluation.benchmark import count_parameters
from sr_pipeline.models.espcn import ESPCN


@pytest.fixture
def lr_rgb() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 12, 17, generator=generator)


@pytest.mark.parametrize("height, width", [(12, 17), (1, 1), (45, 80)])
def test_baseline_output_is_rgb_x4(height: int, width: int) -> None:
    lr = torch.rand(2, 3, height, width)
    sr = ESPCN(scale=4)(lr)
    assert sr.shape == (2, 3, 4 * height, 4 * width)
    assert sr.dtype == lr.dtype
    assert torch.isfinite(sr).all()


def test_forward_raw_returns_sr_rgb_matching_forward(lr_rgb: torch.Tensor) -> None:
    model = ESPCN().eval()
    with torch.no_grad():
        raw = model.forward_raw(lr_rgb)
        assert set(raw) == {"sr_rgb"}
        assert torch.equal(raw["sr_rgb"], model(lr_rgb))


def test_parameter_counts_match_spec() -> None:
    """SPEC: RGB 약 37.2K, B(Y 1채널) 약 24.8K."""
    assert count_parameters(ESPCN()) == 37_200
    assert count_parameters(ESPCN(use_y_channel=True)) == 24_752


def test_structure_tanh_hidden_and_linear_last() -> None:
    model = ESPCN()
    activations = [m for m in model.modules() if not isinstance(m, (nn.Conv2d, nn.Sequential, ESPCN, nn.PixelShuffle))]
    assert len(activations) == 2 and all(isinstance(m, nn.Tanh) for m in activations)
    # 마지막 층은 활성화 없음: 출력이 tanh 범위 [-1, 1]에 묶이지 않는다.
    with torch.no_grad():
        model.last.bias.fill_(2.0)
        assert (model(torch.rand(1, 3, 4, 4)) > 1.0).all()


def test_output_is_not_clipped_to_unit_range() -> None:
    """손실은 자르지 않은 출력으로 계산하므로 모델은 [0, 1]로 자르지 않는다."""
    model = ESPCN()
    with torch.no_grad():
        model.last.bias.fill_(-0.5)
        assert (model(torch.rand(1, 3, 4, 4)) < 0.0).all()


def test_same_seed_gives_same_initialization() -> None:
    torch.manual_seed(123)
    first = ESPCN().state_dict()
    torch.manual_seed(123)
    second = ESPCN().state_dict()
    torch.manual_seed(456)
    third = ESPCN().state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not all(torch.equal(first[k], third[k]) for k in first)


def test_gradients_flow_to_all_parameters(lr_rgb: torch.Tensor) -> None:
    model = ESPCN()
    model(lr_rgb).abs().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError):
        ESPCN()(torch.rand(1, 1, 8, 8))
    with pytest.raises(ValueError):
        ESPCN()(torch.rand(3, 8, 8))


def test_runs_on_given_device() -> None:
    device = torch.device("cpu")
    model = ESPCN().to(device)
    assert model(torch.rand(1, 3, 6, 6, device=device)).device == device
