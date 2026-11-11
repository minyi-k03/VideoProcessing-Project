"""main 병합 실행 기록 자동 집계 CLI 진입점 (SPEC.md)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate merged run records into tables and figures.")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"), help="Directory of merged run records.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/aggregated"), help="Aggregated outputs dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"TODO: Run export_report_assets with {args}.")


if __name__ == "__main__":
    main()
