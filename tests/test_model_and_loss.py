"""모델 및 손실 스위치(A, B, C, B′) 바깥 동작 테스트 (SPEC.md Testing Decisions §1, Task T3.4·T3.5).

모델 내부 층에 접근하지 않고 `forward_raw`/`forward`, 파라미터, `SRLoss`의 반환값·기울기만 검증한다.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from sr_pipeline.common.bicubic import bicubic_resize
from sr_pipeline.common.color import denormalize_y, normalize_y, rgb_to_y, rgb_to_ycbcr, ycbcr_to_rgb
from sr_pipeline.common.seed import set_model_seed
from sr_pipeline.evaluation.benchmark import count_parameters
from sr_pipeline.losses.sr_loss import SRLoss
from sr_pipeline.models.espcn import ESPCN


pytestmark = pytest.mark.usefixtures("restore_global_random_state")


def _rgb(batch: int = 2, height: int = 6, width: int = 7, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, height, width, generator=generator)


def _zero_all_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()


def _step_edge_rgb(size: int = 16) -> torch.Tensor:
    image = torch.full((1, 3, size, size), 0.2)
    image[..., size // 2 :] = 0.8
    return image


# ---------------------------------------------------------------- A: 에지 손실


def test_switch_a_zero_lambda_matches_baseline_and_flat_finite() -> None:
    """A: lambda=0이면 베이스라인 손실과 같고, 평탄 영상에서도 손실·기울기가 유한해야 합니다."""
    hr = _rgb()
    sr = {"sr_rgb": hr + 0.1 * torch.randn_like(hr)}
    assert SRLoss(use_edge_loss=True, edge_loss_lambda=0.0)(sr, hr) == SRLoss()(sr, hr)

    flat_hr = torch.full((1, 3, 8, 8), 0.5)
    flat_sr = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)  # 차이 0: √를 썼다면 기울기가 발산한다
    loss = SRLoss(use_edge_loss=True, edge_loss_lambda=1.0)({"sr_rgb": flat_sr}, flat_hr)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flat_sr.grad).all()


def test_switch_a_penalizes_blurred_edges() -> None:
    hr = _step_edge_rgb()
    blurred = F.avg_pool2d(hr, 5, stride=1, padding=2, count_include_pad=False)
    base = SRLoss()({"sr_rgb": blurred}, hr)
    with_edge = SRLoss(use_edge_loss=True, edge_loss_lambda=0.1)({"sr_rgb": blurred}, hr)
    assert with_edge > base
    # 균일한 오프셋만 다르고 에지가 같으면 에지 항은 0이다.
    shifted = hr + 0.05
    assert SRLoss(use_edge_loss=True, edge_loss_lambda=1.0)({"sr_rgb": shifted}, hr).item() == pytest.approx(0.05)


@pytest.mark.parametrize("lam", [-0.1, math.nan, math.inf])
def test_switch_a_rejects_invalid_lambda(lam: float) -> None:
    with pytest.raises(ValueError):
        SRLoss(use_edge_loss=True, edge_loss_lambda=lam)


def test_lambda_is_ignored_without_switch_a() -> None:
    """T3.7이 공통 λ를 모든 구성에 넘겨도 A가 꺼진 구성은 영향을 받지 않는다."""
    hr = _rgb()
    sr = {"sr_rgb": hr + 0.1}
    assert SRLoss(edge_loss_lambda=0.5)(sr, hr) == SRLoss()(sr, hr)


# ---------------------------------------------------------------- B: Y 채널 처리


def test_switch_b_outputs_rgb_fewer_params_and_matching_loss_scale() -> None:
    """B: SR 결과는 RGB 3채널이며 파라미터 수가 감소하고 손실 크기가 RGB 손실 범위와 맞아야 합니다."""
    lr = _rgb()
    model = ESPCN(use_y_channel=True)
    raw = model.forward_raw(lr)
    assert raw["sr_rgb"].shape == (2, 3, 24, 28)
    assert raw["sr_y_norm"].shape == (2, 1, 24, 28)
    assert count_parameters(model) == 24_752 < count_parameters(ESPCN()) == 37_200

    # 회색 0.1 오차: RGB L1 = 0.1, 정규화된 Y L1도 0.1 (같은 범위)
    hr = _rgb(height=24, width=28) * 0.8 + 0.1
    rgb_loss = SRLoss()({"sr_rgb": hr + 0.1}, hr)
    y_loss = SRLoss(use_y_loss=True)({"sr_rgb": hr + 0.1, "sr_y_norm": normalize_y(rgb_to_y(hr + 0.1))}, hr)
    assert rgb_loss.item() == pytest.approx(0.1, abs=1e-6)
    assert y_loss.item() == pytest.approx(0.1, abs=1e-5)


def test_switch_b_rgb_result_carries_restored_y_and_bicubic_chroma() -> None:
    lr = _rgb()
    raw = ESPCN(use_y_channel=True).forward_raw(lr)
    sr_ycbcr = rgb_to_ycbcr(raw["sr_rgb"])
    assert torch.allclose(sr_ycbcr[:, :1], denormalize_y(raw["sr_y_norm"]), atol=1e-3)
    assert torch.allclose(sr_ycbcr[:, 1:], bicubic_resize(rgb_to_ycbcr(lr)[:, 1:], 4.0), atol=1e-3)


def test_switch_b_keeps_gray_input_gray() -> None:
    gray = _rgb(1)[:, :1].expand(1, 3, 6, 7).contiguous()
    sr = ESPCN(use_y_channel=True)(gray)
    assert torch.allclose(sr[:, 0], sr[:, 1], atol=1e-4) and torch.allclose(sr[:, 1], sr[:, 2], atol=1e-4)


def test_switch_b_output_requires_y_loss() -> None:
    """B 모델의 결과에 RGB 손실을 붙이면(배선 실수) 학습 전에 실패한다."""
    lr, hr = _rgb(), _rgb(height=24, width=28, seed=1)
    with pytest.raises(ValueError):
        SRLoss()(ESPCN(use_y_channel=True).forward_raw(lr), hr)


def test_switch_b_rejects_non_float_y() -> None:
    hr = _rgb(height=8, width=8)
    outputs = {"sr_rgb": hr.clone(), "sr_y_norm": torch.zeros(2, 1, 8, 8, dtype=torch.uint8)}
    with pytest.raises(ValueError):
        SRLoss(use_y_loss=True)(outputs, hr)


def test_switch_b_loss_uses_model_y() -> None:
    """B 손실은 모델의 sr_y_norm으로 계산한다(학습 채널 = 손실 채널)."""
    hr = _rgb(height=8, width=8)
    target_y = normalize_y(rgb_to_y(hr))
    outputs = {"sr_rgb": hr.clone(), "sr_y_norm": target_y + 0.2}  # RGB는 같고 Y만 0.2 다름
    assert SRLoss(use_y_loss=True)(outputs, hr).item() == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------- C: 전역 잔차


def test_switch_c_zero_residual_equals_bicubic_and_allows_negative() -> None:
    """C: 예측 차이가 0이면 bicubic 확대 결과와 같고 음수 차이도 출력에 반영되어야 합니다."""
    lr = _rgb()
    model = ESPCN(use_global_residual=True)
    _zero_all_parameters(model)
    assert torch.allclose(model(lr), bicubic_resize(lr, 4.0), atol=1e-6)

    # 같은 시드로 만든 C와 베이스라인은 네트워크가 같다: C − bicubic = 베이스라인 예측(음수 포함)
    set_model_seed(0)
    baseline = ESPCN()
    set_model_seed(0)
    residual_model = ESPCN(use_global_residual=True)
    with torch.no_grad():
        prediction = baseline(lr)
        difference = residual_model(lr) - bicubic_resize(lr, 4.0)
    assert (prediction < 0).any()
    assert torch.allclose(difference, prediction, atol=1e-6)


def test_switch_bc_adds_bicubic_in_normalized_y() -> None:
    lr = _rgb()
    model = ESPCN(use_y_channel=True, use_global_residual=True)
    _zero_all_parameters(model)
    raw = model.forward_raw(lr)
    expected_y = bicubic_resize(normalize_y(rgb_to_y(lr)), 4.0)
    assert torch.allclose(raw["sr_y_norm"], expected_y, atol=1e-6)
    # 예측 차이 0인 B+C의 RGB 결과는 YCbCr 각각 bicubic 확대한 결과와 같다.
    assert torch.allclose(raw["sr_rgb"], ycbcr_to_rgb(bicubic_resize(rgb_to_ycbcr(lr), 4.0)), atol=1e-5)


# ---------------------------------------------------------------- B′: Y 손실 통제


def test_switch_bp_matches_baseline_params_and_io_with_y_loss() -> None:
    """B′: 파라미터 수와 입출력은 베이스라인과 같고 손실만 정규화된 Y 채널 기준이어야 합니다."""
    lr = _rgb()
    raw = ESPCN().forward_raw(lr)  # B′ 모델 = 베이스라인 모델
    assert set(raw) == {"sr_rgb"} and raw["sr_rgb"].shape == (2, 3, 24, 28)

    hr = _rgb(height=8, width=8) * 0.5 + 0.25
    ycbcr = rgb_to_ycbcr(hr)
    ycbcr[:, 1] += 10.0  # Cb만 바꾼 SR 결과: Y는 같다
    chroma_only = {"sr_rgb": ycbcr_to_rgb(ycbcr)}
    assert SRLoss(use_y_loss=True)(chroma_only, hr).item() == pytest.approx(0.0, abs=1e-6)
    assert SRLoss()(chroma_only, hr).item() > 0.01

    shifted_rgb = {"sr_rgb": hr + 0.1}
    assert SRLoss(use_y_loss=True)(shifted_rgb, hr).item() == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------- 모든 조합


_COMBINATIONS = [
    pytest.param(a, b, c, b or bp, id=f"A{int(a)}B{int(b)}C{int(c)}Bp{int(bp)}")
    for (a, b, c), bp in itertools.chain(
        ((combo, False) for combo in itertools.product([False, True], repeat=3)), [((False, False, False), True)]
    )
]


@pytest.mark.parametrize("use_a, use_b, use_c, use_y_loss", _COMBINATIONS)
def test_every_switch_combination_trains(use_a: bool, use_b: bool, use_c: bool, use_y_loss: bool) -> None:
    """A·B·C를 독립적으로 켜고 끌 수 있고(US28), 어떤 조합도 RGB → RGB이며 학습 기울기가 유한하다."""
    model = ESPCN(use_y_channel=use_b, use_global_residual=use_c)
    criterion = SRLoss(use_edge_loss=use_a, use_y_loss=use_y_loss, edge_loss_lambda=0.1)
    lr, hr = _rgb(), _rgb(height=24, width=28, seed=1)
    outputs = model.forward_raw(lr)
    assert outputs["sr_rgb"].shape == hr.shape
    loss = criterion(outputs, hr)
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
