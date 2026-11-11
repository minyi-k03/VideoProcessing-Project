"""주 경계(구성 + 시드 -> 실행 기록) 종단간 CPU 테스트 (SPEC.md Testing Decisions §1)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside engine.trainer and evaluation.evaluator")
def test_all_configs_run_end_to_end_on_tiny_synthetic_data() -> None:
    """모든 구성이 작은 합성 데이터·1에폭에서 완주하고 형식에 맞는 summary/per_image 행을 생성해야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside engine.trainer")
def test_same_config_and_seed_reproducible() -> None:
    """같은 구성·시드를 두 번 실행하면 동일한 지표를 얻어야 합니다."""
