"""베이스라인 ESPCN 바깥 동작 테스트 (SPEC.md US21·75, ADR-0002, Task T2.1).

모델 내부 층·속성에 접근하지 않고 `forward`/`forward_raw`/파라미터로 보이는 동작만 검증한다.
합성 데이터만 쓰므로 로컬 CPU에서 돈다. 스위치 B·C 동작은 `test_model_and_loss.py`(T3.4)에서 검증한다.
"""

from __future__ import annotations

import pytest
import torch

from sr_pipeline.common.seed import set_model_seed
from sr_pipeline.evaluation.benchmark import count_parameters
from sr_pipeline.models.espcn import ESPCN


pytestmark = pytest.mark.usefixtures("restore_global_random_state")


def _synthetic_lr(batch: int = 2, height: int = 12, width: int = 17) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch, 3, height, width, generator=generator)


def _fit_constant(target: float, steps: int = 200) -> torch.Tensor:
    """출력이 상수 `target`이 되도록 짧게 학습한 뒤의 출력을 돌려준다(바깥 인터페이스만 사용)."""
    set_model_seed(0)
    model = ESPCN()
    lr = _synthetic_lr(1, 6, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        ((model(lr) - target) ** 2).mean().backward()
        optimizer.step()
    with torch.no_grad():
        return model(lr)


@pytest.mark.parametrize("height, width", [(12, 17), (1, 1), (45, 80)])
def test_baseline_output_is_rgb_x4(height: int, width: int) -> None:
    lr = _synthetic_lr(2, height, width)
    sr = ESPCN()(lr)
    assert sr.shape == (2, 3, 4 * height, 4 * width)
    assert sr.dtype == lr.dtype
    assert torch.isfinite(sr).all()


def test_forward_raw_returns_sr_rgb_matching_forward() -> None:
    lr = _synthetic_lr()
    model = ESPCN().eval()
    with torch.no_grad():
        raw = model.forward_raw(lr)
        assert set(raw) == {"sr_rgb"}
        assert torch.equal(raw["sr_rgb"], model(lr))


def test_parameter_counts_match_spec() -> None:
    """SPEC: RGB 약 37.2K, B(Y 1채널) 약 24.8K."""
    assert count_parameters(ESPCN()) == 37_200
    assert count_parameters(ESPCN(use_y_channel=True)) == 24_752


@pytest.mark.parametrize("target", [1.5, -0.5])
def test_output_can_leave_unit_range(target: float) -> None:
    """마지막 층에 활성화가 없고 [0, 1]로 자르지 않으므로, 학습하면 출력이 [0, 1] 밖으로 나갈 수 있다.

    마지막 층에 tanh·sigmoid가 있거나 출력을 자르면 1.5나 -0.5에 도달할 수 없다.
    """
    sr = _fit_constant(target)
    assert (sr > 1.0).all() if target > 1.0 else (sr < 0.0).all()
    assert abs(sr.mean().item() - target) < 0.05


def test_same_seed_gives_same_initialization() -> None:
    lr = _synthetic_lr()
    outputs = {}
    for name, seed in [("first", 0), ("same", 0), ("other", 1)]:
        set_model_seed(seed)
        with torch.no_grad():
            outputs[name] = ESPCN()(lr)
    assert torch.equal(outputs["first"], outputs["same"])
    assert not torch.equal(outputs["first"], outputs["other"])


def test_gradients_flow_to_all_parameters() -> None:
    model = ESPCN()
    model(_synthetic_lr()).abs().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
    "lr",
    [torch.rand(1, 1, 8, 8), torch.rand(3, 8, 8), torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)],
    ids=["one-channel", "no-batch-dim", "uint8"],
)
def test_rejects_invalid_input(lr: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        ESPCN()(lr)


@pytest.mark.parametrize("scale", [2, 3])
def test_rejects_scale_other_than_x4(scale: int) -> None:
    with pytest.raises(ValueError):
        ESPCN(scale=scale)


def test_runs_on_given_device() -> None:
    device = torch.device("cpu")
    model = ESPCN().to(device)
    assert model(torch.rand(1, 3, 6, 6, device=device)).device == device
