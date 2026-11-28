"""main 병합 실행 기록 자동 집계 CLI 진입점 (SPEC.md US58, ADR-0006, Task T6.3).

기본 실행에서는 `results/runs/`가 git `main` 브랜치와 정확히 일치(미병합·미추적·누락 없음)하는지
검증한 뒤 `results/aggregated/`에 표·그림을 생성합니다.
병합 전 미리보기/테스트용 `--allow-unmerged`를 켜면 기본 공식 출력 폴더(`results/aggregated/`)로의
저장을 거부하고 별도 `--output-dir`만 허용합니다.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from sr_pipeline.reporting.aggregate import export_report_assets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "aggregated"
DEFAULT_FAIR_CONFIG = REPO_ROOT / "configs" / "fair_comparison.yaml"
DEFAULT_EXPERIMENTS_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate merged run records into tables and figures.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Directory of merged run records (default: results/runs).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Aggregated outputs dir (default: results/aggregated).")
    parser.add_argument("--fair-config", type=Path, default=None, help="Path to fair_comparison.yaml.")
    parser.add_argument("--experiments-config", type=Path, default=None, help="Path to experiments.yaml.")
    parser.add_argument("--main-ref", type=str, default="main", help="Git ref representing merged records (default: main).")
    parser.add_argument(
        "--allow-unmerged",
        action="store_true",
        help="Skip main-branch verification for local preview (requires a non-default --output-dir).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional prepared data dir for case-study images.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint dir for case-study images.")
    return parser.parse_args(argv)


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git을 찾을 수 없어 main 병합 여부를 검증할 수 없습니다.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} 실패: {detail}") from error
    return result.stdout.strip()


def verify_runs_merged_to_main(runs_dir: Path, *, main_ref: str = "main", repo_root: Path = REPO_ROOT) -> None:
    """`runs_dir`의 모든 실행 기록이 `main_ref` 브랜치와 100% 일치하는지 검증합니다."""
    resolved_runs = Path(runs_dir).resolve()
    repo_top = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    try:
        rel_runs = resolved_runs.relative_to(repo_top).as_posix()
    except ValueError as error:
        raise RuntimeError(
            f"실행 기록 폴더({resolved_runs})가 git 저장소 밖에 있어 {main_ref} 병합 여부를 확인할 수 없습니다. "
            "미리보기용이면 --allow-unmerged와 별도 --output-dir을 지정하세요."
        ) from error

    _git(repo_top, "rev-parse", "--verify", main_ref)
    dirty = _git(repo_top, "status", "--porcelain", "--untracked-files=all", "--", rel_runs)
    if dirty:
        raise RuntimeError(
            f"{rel_runs}에 커밋되지 않았거나 추적되지 않은 실행 기록이 있습니다. "
            f"커밋 및 {main_ref} 병합 후 집계하세요:\n{dirty}"
        )
    diff = _git(repo_top, "diff", "--name-status", main_ref, "--", rel_runs)
    if diff:
        raise RuntimeError(
            f"{rel_runs}가 {main_ref} 브랜치에 병합된 실행 기록과 다릅니다"
            f"(미병합 실행이 있거나 {main_ref} 최신 병합 기록이 빠져 있습니다):\n{diff}"
        )


def _check_output_target(
    output_dir: Path,
    runs_dir: Path,
    fair_config: Path,
    experiments_config: Path,
    allow_unmerged: bool,
    main_ref: str,
) -> None:
    """공식 출력 폴더(`results/aggregated/`)에는 main 병합 검증과 기본 설정을 거친 집계만 허용한다."""
    default_out = DEFAULT_OUTPUT_DIR.resolve()
    resolved_out = Path(output_dir).resolve()
    is_default_target = resolved_out == default_out or default_out in resolved_out.parents
    if not is_default_target:
        return
    if allow_unmerged or main_ref != "main":
        raise ValueError(
            "공식 집계 폴더(results/aggregated/)에는 main 기준 검증을 통과한 집계만 쓸 수 있습니다"
            "(--allow-unmerged 또는 --main-ref 변경 시 별도 --output-dir을 지정하세요)."
        )
    if (
        Path(runs_dir).resolve() != DEFAULT_RUNS_DIR.resolve()
        or Path(fair_config).resolve() != DEFAULT_FAIR_CONFIG.resolve()
        or Path(experiments_config).resolve() != DEFAULT_EXPERIMENTS_CONFIG.resolve()
    ):
        raise ValueError(
            "공식 집계 폴더(results/aggregated/)에는 기본 results/runs 및 configs/ 설정을 쓴 집계만 저장할 수 있습니다. "
            "다른 경로·설정을 쓰려면 별도 --output-dir을 지정하세요."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir = args.runs_dir if args.runs_dir is not None else DEFAULT_RUNS_DIR
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    fair_config = args.fair_config if args.fair_config is not None else DEFAULT_FAIR_CONFIG
    experiments_config = args.experiments_config if args.experiments_config is not None else DEFAULT_EXPERIMENTS_CONFIG

    _check_output_target(output_dir, runs_dir, fair_config, experiments_config, args.allow_unmerged, args.main_ref)
    if not args.allow_unmerged:
        verify_runs_merged_to_main(runs_dir, main_ref=args.main_ref, repo_root=REPO_ROOT)

    manifest = export_report_assets(
        runs_dir,
        output_dir,
        fair_config_path=fair_config,
        experiments_config_path=experiments_config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"aggregated {len(manifest['run_ids'])} runs -> {len(manifest['tables'])} tables, {len(manifest['figures'])} figures")


if __name__ == "__main__":
    main()
