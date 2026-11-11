"""발표용 3분할 라이브 데모 CLI 진입점 (SPEC.md / User Stories 79~81)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 3-way live demo (Bicubic / Baseline / Proposed).")
    parser.add_argument("--baseline-ckpt", type=Path, required=True, help="Baseline checkpoint path.")
    parser.add_argument("--proposed-ckpt", type=Path, required=True, help="Proposed config checkpoint path.")
    parser.add_argument("--proposed-config-id", type=str, required=True, help="Proposed config ID.")
    parser.add_argument("--sample-dir", type=Path, required=True, help="Representative/upload LR image directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"), help="Demo visualization output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"TODO: Launch live demo with {args}.")


if __name__ == "__main__":
    main()
