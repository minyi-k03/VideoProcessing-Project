"""통합 평가 모듈 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside evaluation.metrics")
def test_identical_images_psnr_inf_and_ssim_one() -> None:
    """같은 영상이면 PSNR(Y)은 무한대, SSIM(Y)은 1이어야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside evaluation.postprocess and evaluation.metrics")
def test_border_4px_difference_does_not_affect_metrics() -> None:
    """가장자리 4화소 안쪽의 차이는 평가 지표에 영향을 주지 않아야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside evaluation.metrics")
def test_edge_psnr_defined_on_completely_flat_image() -> None:
    """완전히 평탄한 영상에서도 에지 PSNR이 정의되어야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside evaluation.postprocess")
def test_clamp_and_quantize_order_consistent_across_configs() -> None:
    """자르기([0,1]) -> 8비트 양자화 -> 가장자리 4화소 잘라내기 순서가 일관되어야 합니다."""
