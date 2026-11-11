"""공정 비교 조건 및 구성 스위치 해석·검증 (SPEC.md / ADR-0001).

구성은 허용된 개선 스위치(A, B, C, Bp)와 시드 외의 키를 가질 수 없으며,
공정 비교 조건을 덮어쓰려 하면 실행 시작 전에 거부합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Switches:
    """개선 스위치 조합."""

    A: bool = False
    B: bool = False
    C: bool = False
    Bp: bool = False


@dataclass(frozen=True)
class ResolvedRunConfig:
    """공정 비교 조건 + 구성 스위치 + 시드가 결합된 단일 실행 조건."""

    config_id: str
    seed: int
    trainable: bool
    switches: Switches
    fair_comparison: dict[str, Any]


def load_fair_comparison(path: Path) -> dict[str, Any]:
    """`configs/fair_comparison.yaml`을 로드합니다."""
    raise NotImplementedError("TODO: Load and validate fair_comparison.yaml.")


def load_experiment_config(path: Path, config_id: str) -> tuple[bool, Switches]:
    """`configs/experiments.yaml`에서 구성 정의를 로드하고 허용되지 않은 키를 거부합니다."""
    raise NotImplementedError("TODO: Load experiment config and reject any fair-comparison override.")


def resolve_run_config(
    fair_path: Path,
    experiments_path: Path,
    config_id: str,
    seed: int,
) -> ResolvedRunConfig:
    """단일 실행에 사용할 최종 실행 조건을 확정합니다."""
    raise NotImplementedError("TODO: Combine fair comparison rules with switch-only config and seed.")
