"""모델 및 손실 스위치(A, B, C, B′) 바깥 동작 테스트 (SPEC.md Testing Decisions §1)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside losses.sr_loss")
def test_switch_a_zero_lambda_matches_baseline_and_flat_finite() -> None:
    """A: lambda=0이면 베이스라인 손실과 같고, 평탄 영상에서도 손실·기울기가 유한해야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside models.espcn and losses.sr_loss")
def test_switch_b_outputs_rgb_fewer_params_and_matching_loss_scale() -> None:
    """B: SR 결과는 RGB 3채널이며 파라미터 수가 감소하고 손실 크기가 RGB 손실 범위와 맞아야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside models.espcn")
def test_switch_c_zero_residual_equals_bicubic_and_allows_negative() -> None:
    """C: 예측 차이가 0이면 bicubic 확대 결과와 같고 음수 차이도 출력에 반영되어야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside models.espcn and losses.sr_loss")
def test_switch_bp_matches_baseline_params_and_io_with_y_loss() -> None:
    """B′: 파라미터 수와 입출력은 베이스라인과 같고 손실만 정규화된 Y 채널 기준이어야 합니다."""
