"""구성+시드 단일 실행 파이프라인 (SPEC.md / ADR-0001 / ADR-0005).

실행 1회가 결과 수치의 최소 출처 단위입니다:
- run_id = <commit>-<config_id>-s<seed>
- 커밋되지 않은 변경 감지(경고 또는 거부)
- 에폭 기반 학습 (3시간 상한 초과 시 incomplete 기록)
- 중간/마지막 에폭 체크포인트 저장 및 동일 run_id 재개 지원
- 마지막 에폭(평가 체크포인트)으로 통합 평가 실행 및 실행 기록 저장
"""

from __future__ import annotations

from pathlib import Path

from sr_pipeline.config.schema import ResolvedRunConfig


def build_run_id(commit: str, config_id: str, seed: int) -> str:
    """SPEC.md 형식의 실행 ID(`<commit>-<config_id>-s<seed>`)를 생성합니다."""
    return f"{commit}-{config_id}-s{seed}"


def check_git_clean(allow_dirty: bool = False) -> str:
    """현재 HEAD 커밋 해시를 반환하며, 커밋되지 않은 변경이 있으면 경고하거나 예외를 발생시킵니다."""
    raise NotImplementedError("TODO: Inspect git status/HEAD commit and guard against dirty state.")


def train_and_evaluate(
    run_config: ResolvedRunConfig,
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    resume_run_id: str | None = None,
    allow_dirty: bool = False,
) -> Path:
    """학습부터 통합 평가 및 실행 기록 저장까지 1회 실행을 완결합니다."""
    raise NotImplementedError("TODO: Implement epoch training loop with 3h safety cap, checkpointing, and eval.")
