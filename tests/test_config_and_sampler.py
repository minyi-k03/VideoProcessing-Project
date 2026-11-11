"""공정 비교 조건 검증 및 시드 기반 샘플러 테스트 (SPEC.md Testing Decisions §1)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside config.schema")
def test_reject_fair_comparison_override() -> None:
    """구성이 공정 비교 조건을 덮어쓰려 하면 실행 전에 거부되어야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside config.schema")
def test_resolved_configs_differ_only_by_switches_and_seed() -> None:
    """서로 다른 구성의 확정 실행 조건은 스위치와 시드 외에 차이가 없어야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside data.sampler")
def test_sampler_same_order_for_same_seed_across_configs() -> None:
    """같은 시드면 구성이 달라도 샘플러가 동일한 패치 순서를 생성해야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside data.sampler")
def test_sampler_different_order_for_different_seeds() -> None:
    """시드가 다르면 다른 패치 순서를 생성해야 합니다."""
