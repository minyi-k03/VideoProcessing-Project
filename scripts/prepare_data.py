"""DIV2K 패치 풀 및 테스트셋 LR 생성 CLI 진입점 (SPEC.md / ADR-0001)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared patch pool and x4 testset LR images.")
    parser.add_argument("--div2k-hr-dir", type=Path, required=True, help="DIV2K train HR directory.")
    parser.add_argument("--testsets-hr-dir", type=Path, required=True, help="Testsets HR root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output data directory.")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (fixed: 4).")
    parser.add_argument("--patch-size-hr", type=int, default=96, help="HR patch size.")
    parser.add_argument("--num-patches", type=int, default=10000, help="Total patches in shared pool.")
    parser.add_argument("--pool-seed", type=int, default=42, help="Deterministic seed for patch pool creation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"TODO: Invoke create_patch_pool and prepare_testsets with {args}.")


if __name__ == "__main__":
    main()
