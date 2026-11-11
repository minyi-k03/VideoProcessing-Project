"""단일 실행(구성 + 시드 -> 학습 + 통합 평가 -> 실행 기록) CLI 진입점 (SPEC.md)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single controlled x4 SR experiment.")
    parser.add_argument("--config", type=str, required=True, help="Config ID (e.g., bicubic, baseline, A, B, AB, C, Bp).")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True, help="Run seed in {0, 1, 2}.")
    parser.add_argument("--fair-config", type=Path, default=Path("configs/fair_comparison.yaml"))
    parser.add_argument("--experiments-config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Runtime local data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"), help="Run records output dir.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint storage dir (Drive or local).")
    parser.add_argument("--resume-run-id", type=str, default=None, help="Optional run_id to resume.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree (logs warning).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"TODO: Resolve config and execute train_and_evaluate with {args}.")


if __name__ == "__main__":
    main()
