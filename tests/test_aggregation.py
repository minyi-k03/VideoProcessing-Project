"""자동 집계 보조 경계 테스트 (SPEC.md Testing Decisions §3)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside reporting.aggregate")
def test_aggregation_mean_std_delta_and_sign_consistency() -> None:
    """고정된 가짜 실행 기록 입력으로 평균 ± 표준편차, Δ, 시드별 Δ 부호 일관성이 기대값과 같아야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside reporting.aggregate")
def test_aggregation_marks_incomplete_and_rejects_duplicates() -> None:
    """미완료/시드 부족 표시는 정확해야 하며 (config_id, seed, patch_pool_id) 중복 시 오류를 내야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside reporting.aggregate")
def test_per_image_delta_and_case_study_selection() -> None:
    """영상별 Δ 계산과 성공·실패 사례 영상 자동 선정이 규칙대로 동작해야 합니다."""
