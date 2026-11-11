"""1주차 파일럿 베이스라인 학습/진단 4종 CLI 진입점 (SPEC.md / User Story 51)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot baseline diagnosis suite (error map, edge/flat, FFT, YCbCr).")
    parser.add_argument("--sr-dir", type=Path, required=True, help="Pilot SR outputs directory.")
    parser.add_argument("--hr-dir", type=Path, required=True, help="Corresponding HR testsets directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot"), help="Pilot diagnosis output dir.")
    parser.add_argument("--edge-top-q", type=float, default=20.0, help="Top q%% Sobel edge region percentage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"TODO: Run run_diagnosis_suite with {args}.")


if __name__ == "__main__":
    main()
