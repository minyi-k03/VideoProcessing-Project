"""색·보간 공용 규칙 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sr_pipeline.common.color import (
    denormalize_y,
    normalize_y,
    rgb_to_y,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def _matlab_rgb2ycbcr_reference(rgb: np.ndarray) -> np.ndarray:
    """MATLAB rgb2ycbcr(double, [0, 1])의 정의를 numpy로 직접 옮긴 참조 구현. rgb: (H, W, 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128.0 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128.0 + 112.0 * r - 93.786 * g - 18.214 * b
    return np.stack([y, cb, cr], axis=-1)


def _random_rgb(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, 8, 9, generator=generator, dtype=torch.float64)


def test_bt601_y_matches_matlab_reference() -> None:
    """BT.601 Y 변환이 MATLAB/BasicSR 참조 구현과 허용 오차 안에서 일치해야 합니다."""
    rgb = _random_rgb()
    expected = _matlab_rgb2ycbcr_reference(rgb[0].permute(1, 2, 0).numpy())

    ycbcr = rgb_to_ycbcr(rgb)[0].permute(1, 2, 0).numpy()
    y_only = rgb_to_y(rgb)[0, 0].numpy()

    np.testing.assert_allclose(ycbcr, expected, atol=1e-9)
    np.testing.assert_allclose(y_only, expected[..., 0], atol=1e-9)


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


def test_color_functions_reject_wrong_channels() -> None:
    """3채널이 아닌 입력은 조용히 넘어가지 않고 오류여야 합니다."""
    with pytest.raises(ValueError):
        rgb_to_ycbcr(torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        ycbcr_to_rgb(torch.zeros(3, 4, 4))


def test_color_runs_on_float32_cpu() -> None:
    """학습·평가에서 쓰는 float32 입력에서도 dtype이 유지되어야 합니다."""
    rgb = _random_rgb(2).float()
    assert rgb_to_ycbcr(rgb).dtype == torch.float32
    assert ycbcr_to_rgb(rgb_to_ycbcr(rgb)).dtype == torch.float32


@pytest.mark.skip(reason="Skeleton test: implement alongside common.bicubic")
def test_bicubic_down_and_up_match_matlab_reference() -> None:
    """bicubic 축소(안티에일리어싱)와 확대(화소 중심 정렬)가 MATLAB 호환 참조 구현과 일치해야 합니다."""
