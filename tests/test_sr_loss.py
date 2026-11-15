"""기본 L1 손실 바깥 동작 테스트 (SPEC.md US22 일부·US30, AGENTS.md, Task T2.2).

`SRLoss`의 입력(`forward_raw` 형식의 dict, HR RGB)과 반환값·기울기만 검증한다.
스위치 A·B·B′ 동작은 `test_model_and_loss.py`에서 검증한다.
"""

from __future__ import annotations

import pytest
import torch

from sr_pipeline.losses.sr_loss import SRLoss
from sr_pipeline.models.espcn import ESPCN


def _hr(batch: int = 2, height: int = 8, width: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch, 3, height, width, generator=generator)


def _loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return SRLoss()({"sr_rgb": sr}, hr)


def test_identical_images_give_zero_loss() -> None:
    hr = _hr()
    assert _loss(hr.clone(), hr).item() == 0.0


def test_constant_offset_gives_offset() -> None:
    hr = _hr()
    assert _loss(hr + 0.1, hr).item() == pytest.approx(0.1, abs=1e-6)
    assert _loss(hr - 0.1, hr).item() == pytest.approx(0.1, abs=1e-6)


def test_loss_is_mean_over_pixels_and_channels() -> None:
    """화소 1개·채널 1개만 0.6 다르면 손실은 0.6 / (B·C·H·W)."""
    hr = torch.zeros(1, 3, 4, 5)
    sr = hr.clone()
    sr[0, 1, 2, 3] = 0.6
    assert _loss(sr, hr).item() == pytest.approx(0.6 / (3 * 4 * 5))


@pytest.mark.parametrize("batch, height, width", [(1, 4, 4), (3, 16, 9)])
def test_same_error_gives_same_loss_for_any_size(batch: int, height: int, width: int) -> None:
    hr = _hr(batch, height, width)
    assert _loss(hr + 0.25, hr).item() == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("sr_value, hr_value", [(1.5, 1.0), (-0.3, 0.0)])
def test_uses_unclipped_output_and_keeps_gradient(sr_value: float, hr_value: float) -> None:
    """[0, 1]로 잘랐다면 손실이 0이고 기울기도 0이다. 자르지 않으므로 둘 다 0이 아니다."""
    sr = torch.full((1, 3, 2, 2), sr_value, requires_grad=True)
    hr = torch.full((1, 3, 2, 2), hr_value)
    loss = _loss(sr, hr)
    assert loss.item() == pytest.approx(abs(sr_value - hr_value))
    loss.backward()
    assert (sr.grad != 0).all()


def test_works_with_espcn_forward_raw() -> None:
    model = ESPCN()
    lr = torch.rand(2, 3, 4, 5)
    hr = torch.rand(2, 3, 16, 20)
    loss = SRLoss()(model.forward_raw(lr), hr)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize(
    "sr, hr",
    [
        (torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 9)),
        (torch.rand(3, 8, 8), torch.rand(3, 8, 8)),
        (torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)),
        (torch.rand(1, 3, 8, 8), torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)),
    ],
    ids=["shape-mismatch", "no-batch-dim", "one-channel", "uint8-hr"],
)
def test_rejects_invalid_input(sr: torch.Tensor, hr: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _loss(sr, hr)
