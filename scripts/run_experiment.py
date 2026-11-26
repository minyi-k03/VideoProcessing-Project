"""단일 실행(구성 + 시드 -> 학습 + 통합 평가 -> 실행 기록) CLI 진입점 (SPEC.md US31, Task T3.8).

Colab(Drive)과 Kaggle에서 경로 인자만 바꿔 같은 명령으로 돈다.
- Colab: `--data-archive <런타임 로컬로 복사한 zip>`을 주면 `--data-dir`로 풀어 쓴다(이미 풀려 있으면 재사용).
- Kaggle: Dataset이 자동 해제돼 붙으므로 `--data-dir /kaggle/input/...`만 준다(복사·쓰기 없음).
순서는 실행 조건 확정(구성·시드·`pilot_determined`의 `null` 검사) → 코드 커밋·재개 요청 확인 → 데이터 준비 → 학습·평가다.
앞의 검사가 실패하면 큰 압축 파일을 풀기 전에 멈춘다. 실행 기록 커밋·push는 노트북 02가 `engine/publish.py`로 한다.
입력·저장 경로는 출력하지 않는다(노트북 출력이 Public 저장소에 커밋될 수 있다).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from sr_pipeline.config.schema import resolve_run_config
from sr_pipeline.engine.runtime_data import extract_data_archive, find_data_root
from sr_pipeline.engine.trainer import check_git_clean, check_resume_request, train_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
# 기본 경로는 실행 위치(CWD)와 상관없이 이 저장소를 가리킨다. 실행 기록 커밋(`engine/publish.py`)이 `<저장소>/results/runs`를 기대한다.
DEFAULT_FAIR_CONFIG = REPO_ROOT / "configs" / "fair_comparison.yaml"
DEFAULT_EXPERIMENTS_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single controlled x4 SR experiment.")
    parser.add_argument("--config", type=str, required=True, help="Config ID (e.g., bicubic, baseline, A, B, AB, C, Bp).")
    parser.add_argument("--seed", type=int, required=True, help="Run seed (must be in fixed_rules.seeds).")
    parser.add_argument("--fair-config", type=Path, default=DEFAULT_FAIR_CONFIG)
    parser.add_argument("--experiments-config", type=Path, default=DEFAULT_EXPERIMENTS_CONFIG)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with patch_pool/ and testsets/ (Kaggle: the attached input folder; Colab: where --data-archive is unpacked).",
    )
    parser.add_argument("--data-archive", type=Path, default=None, help="Data zip to unpack into --data-dir (Colab/Drive).")
    parser.add_argument("--archive-sha256", type=str, default=None, help="Expected sha256 of --data-archive.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Run records output dir.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint storage dir (Drive or local).")
    parser.add_argument("--resume-run-id", type=str, default=None, help="Optional run_id to resume.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree (logs warning, run_id gets -dirty).")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training/evaluation device."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_config = resolve_run_config(args.fair_config, args.experiments_config, args.config, args.seed)
    # 압축 파일을 풀기 전에 빠르게 거부: 커밋 안 된 코드 변경, 이어 갈 수 없는 재개 요청
    commit = check_git_clean(REPO_ROOT, allow_dirty=args.allow_dirty)
    check_resume_request(run_config, commit, args.resume_run_id)

    if args.data_archive is not None:
        extract_data_archive(args.data_archive, args.data_dir, expected_sha256=args.archive_sha256)
    data_root = find_data_root(args.data_dir)

    run_dir = train_and_evaluate(
        run_config,
        data_root,
        args.output_dir,
        args.checkpoint_dir,
        resume_run_id=args.resume_run_id,
        allow_dirty=args.allow_dirty,
        repo_root=REPO_ROOT,
        device=args.device,
    )
    summary = pd.read_csv(run_dir / "summary.csv")
    print(f"run_id: {run_dir.name}")
    print(f"status: {', '.join(sorted(set(summary['status'])))}")
    print("records: <output-dir>/<run_id> (summary.csv, per_image.csv" + (", train_log.json" if run_config.trainable else "") + ")")


if __name__ == "__main__":
    main()
