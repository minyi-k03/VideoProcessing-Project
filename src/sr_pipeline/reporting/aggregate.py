"""실행 기록 자동 집계 및 보고서/발표용 표·그림 생성 (SPEC.md / ADR-0003).

원칙:
- main에 병합된 실행 기록(`results/runs/`)만 입력으로 사용
- `(config_id, seed, patch_pool_id)` 중복 감지 시 오류
- 시드 3개 미만 또는 `incomplete` 포함 구성은 "미완료" 표시
- 평균 ± 표준편차, 베이스라인 대비 Δ, 시드별 Δ 부호 일관성, |mean Δ| > std(seed Δ) 판정
- Ablation 표(A: RGB·Y 맥락, B: 정렬 B′·입력/용량 B 분해), 영상별 Δ 분포, 사례 영상 자동 선정
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_run_records(runs_dir: Path) -> tuple[Any, Any]:
    """`runs_dir` 하위의 summary 및 per_image 기록을 로드합니다."""
    raise NotImplementedError("TODO: Load summary and per_image run records.")


def validate_unique_runs(summary_df: Any) -> None:
    """`(config_id, seed, patch_pool_id)` 조합 중복 여부를 검사합니다."""
    raise NotImplementedError("TODO: Raise error on duplicate (config_id, seed, patch_pool_id) records.")


def compute_ablation_tables(summary_df: Any) -> dict[str, Any]:
    """결과 표, Ablation 표, 시드별 Δ 일관성, 학습 시간 표를 생성합니다."""
    raise NotImplementedError("TODO: Aggregate mean+/-std, deltas, seed-paired deltas, and completeness.")


def select_case_study_images(per_image_df: Any, config_id: str) -> dict[str, Any]:
    """영상별 Δ의 상위·하위에서 성공·실패 사례 영상을 규칙으로 자동 선정합니다."""
    raise NotImplementedError("TODO: Select top/bottom per-image delta cases objectively.")


def export_report_assets(runs_dir: Path, output_dir: Path) -> None:
    """표(CSV/Markdown), 분포 그림, 다중 구성 비교 그림, 학습 곡선, 진행 현황 표를 내보냅니다."""
    raise NotImplementedError("TODO: Run full aggregation pipeline and write report assets.")
