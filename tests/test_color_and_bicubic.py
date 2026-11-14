"""색·보간 공용 규칙 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import sr_pipeline.common.color as color_module
from sr_pipeline.common.color import (
    denormalize_y,
    normalize_y,
    rgb_to_y,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def _matlab_rgb2ycbcr_reference(rgb: np.ndarray) -> np.ndarray:
    """BasicSR/MATLAB rgb2ycbcr(uint8 스케일)의 정의를 numpy로 직접 옮긴 참조 구현. rgb: (H, W, 3).

    구현과 같은 계수를 다시 적은 것이라 계수 오타는 못 잡는다. 계수 자체는 아래
    test_bt601_primary_colors_known_values의 널리 알려진 고정 값이 독립적으로 지킨다.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128.0 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128.0 + 112.0 * r - 93.786 * g - 18.214 * b
    return np.stack([y, cb, cr], axis=-1)


def _random_rgb(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, 8, 9, generator=generator, dtype=torch.float64)


@pytest.fixture
def fresh_color_cache():
    """상수 캐시 상태가 테스트끼리 새지 않게 앞뒤로 비운다."""
    color_module._constants.cache_clear()
    yield
    color_module._constants.cache_clear()


def test_bt601_y_matches_matlab_reference() -> None:
    """BT.601 Y 변환이 MATLAB/BasicSR 참조 구현과 허용 오차 안에서 일치해야 합니다(배치 전체)."""
    rgb = _random_rgb()
    ycbcr = rgb_to_ycbcr(rgb)
    y_only = rgb_to_y(rgb)

    for index in range(rgb.shape[0]):
        expected = _matlab_rgb2ycbcr_reference(rgb[index].permute(1, 2, 0).numpy())
        np.testing.assert_allclose(ycbcr[index].permute(1, 2, 0).numpy(), expected, atol=1e-9)
        np.testing.assert_allclose(y_only[index, 0].numpy(), expected[..., 0], atol=1e-9)


@pytest.mark.parametrize(
    ("rgb", "expected_ycbcr"),
    [
        ((1.0, 0.0, 0.0), (81.481, 90.203, 240.0)),
        ((0.0, 1.0, 0.0), (144.553, 53.797, 34.214)),
        ((0.0, 0.0, 1.0), (40.966, 240.0, 109.786)),
        ((1.0, 1.0, 1.0), (235.0, 128.0, 128.0)),
        ((0.0, 0.0, 0.0), (16.0, 128.0, 128.0)),
    ],
)
def test_bt601_primary_colors_known_values(rgb, expected_ycbcr) -> None:
    """원색·흑백의 BT.601 값(적색 81.481/90.203/240 등)이 알려진 고정 값과 일치해야 합니다.

    구현 계수를 다시 쓰지 않은 값이라, 계수 오타가 참조 구현과 같이 틀려도 여기서 잡힌다.
    rgb_to_y도 같은 값(Y)을 내야 한다.
    """
    pixel = torch.tensor(rgb, dtype=torch.float64).view(1, 3, 1, 1)
    assert rgb_to_ycbcr(pixel).flatten().tolist() == pytest.approx(expected_ycbcr, abs=1e-3)
    assert rgb_to_y(pixel).item() == pytest.approx(expected_ycbcr[0], abs=1e-3)


def test_bt601_range_endpoints() -> None:
    """검정은 Y=16, 흰색은 Y=235, 회색의 Cb·Cr은 128이어야 합니다."""
    black = torch.zeros(1, 3, 2, 2)
    white = torch.ones(1, 3, 2, 2)
    gray = torch.full((1, 3, 2, 2), 0.5)

    assert torch.allclose(rgb_to_y(black), torch.full((1, 1, 2, 2), 16.0))
    assert torch.allclose(rgb_to_y(white), torch.full((1, 1, 2, 2), 235.0), atol=1e-4)
    gray_ycbcr = rgb_to_ycbcr(gray)
    assert torch.allclose(gray_ycbcr[:, 1:], torch.full((1, 2, 2, 2), 128.0), atol=1e-4)


def test_rgb_ycbcr_round_trip_does_not_clip() -> None:
    """RGB→YCbCr→RGB 왕복이 원래 값을 복원해야 하고, [0, 1] 밖의 값도 자르지 않아야 합니다."""
    rgb = _random_rgb(1)
    assert torch.allclose(ycbcr_to_rgb(rgb_to_ycbcr(rgb)), rgb, atol=1e-9)

    out_of_range = rgb * 1.5 - 0.25
    restored = ycbcr_to_rgb(rgb_to_ycbcr(out_of_range))
    assert torch.allclose(restored, out_of_range, atol=1e-9)
    assert restored.min() < 0.0 and restored.max() > 1.0


def test_normalize_y_round_trip_and_range() -> None:
    """Y 정규화는 16→0, 235→1로 보내고, 되돌리면 원래 값이어야 합니다."""
    y = torch.tensor([16.0, 125.5, 235.0]).view(1, 1, 1, 3)
    normalized = normalize_y(y)
    assert torch.allclose(normalized[..., 0], torch.tensor(0.0))
    assert torch.allclose(normalized[..., 2], torch.tensor(1.0))
    assert torch.allclose(denormalize_y(normalized), y)


def test_normalize_y_does_not_clip_out_of_range_values() -> None:
    """자르기 전 출력(Y가 16~235 밖)도 정규화가 자르지 않고 [0, 1] 밖 값으로 보존해야 합니다."""
    y = torch.tensor([0.0, 300.0]).view(1, 1, 1, 2)
    normalized = normalize_y(y)
    assert normalized[..., 0] < 0.0 and normalized[..., 1] > 1.0
    assert torch.allclose(denormalize_y(normalized), y, atol=1e-4)  # float32 반올림 오차 허용


def test_color_functions_reject_wrong_channels() -> None:
    """3채널이 아닌 입력은 조용히 넘어가지 않고 오류여야 합니다."""
    with pytest.raises(ValueError):
        rgb_to_ycbcr(torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        rgb_to_y(torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        ycbcr_to_rgb(torch.zeros(3, 4, 4))


def test_color_float32_keeps_dtype_and_accuracy() -> None:
    """본 실행 정밀도(FP32)에서 모든 함수가 dtype을 유지하고 왕복 오차가 작아야 합니다."""
    rgb = _random_rgb(2).float()
    assert rgb_to_ycbcr(rgb).dtype == torch.float32
    assert rgb_to_y(rgb).dtype == torch.float32
    assert normalize_y(rgb_to_y(rgb)).dtype == torch.float32
    assert denormalize_y(normalize_y(rgb_to_y(rgb))).dtype == torch.float32
    restored = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    assert restored.dtype == torch.float32
    assert torch.allclose(restored, rgb, atol=1e-4)


def test_color_gradients_flow_with_expected_values(fresh_color_cache) -> None:
    """B·B′는 손실 그래프 안에서 이 함수들을 부르므로 역전파가 되어야 하고 기울기가 계수와 같아야 합니다."""
    rgb = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    rgb_to_y(rgb).sum().backward()
    expected = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float64).view(1, 3, 1, 1)
    assert torch.allclose(rgb.grad, expected.expand_as(rgb))

    ycbcr = torch.rand(1, 3, 4, 4, requires_grad=True)
    ycbcr_to_rgb(ycbcr).sum().backward()
    assert ycbcr.grad is not None and torch.isfinite(ycbcr.grad).all()


def test_backward_works_after_constants_first_built_in_inference_mode(fresh_color_cache) -> None:
    """평가·추론 시간 측정이 inference_mode로 먼저 돌아도, 이후 학습 역전파가 깨지면 안 됩니다."""
    with torch.inference_mode():
        sample = torch.rand(1, 3, 4, 4)
        rgb_to_ycbcr(sample)
        rgb_to_y(sample)
        ycbcr_to_rgb(sample)

    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    (rgb_to_ycbcr(rgb).sum() + rgb_to_y(rgb).sum() + ycbcr_to_rgb(rgb).sum()).backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()


@pytest.mark.skip(reason="Skeleton test: implement alongside common.bicubic")
def test_bicubic_down_and_up_match_matlab_reference() -> None:
    """bicubic 축소(안티에일리어싱)와 확대(화소 중심 정렬)가 MATLAB 호환 참조 구현과 일치해야 합니다."""
