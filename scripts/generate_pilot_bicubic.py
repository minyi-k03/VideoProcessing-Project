"""파일럿 진단용 Bicubic 하한 SR 결과 생성 CLI 진입점 (Task T2.5 후속).

테스트셋 LR 영상(`<data-dir>/testsets/<데이터셋>/LR/<image_id>.png`)을
`common.bicubic.bicubic_resize(lr, 4.0)`으로 확대하고 `quantize_and_shave(border=0)`로
[0, 1] 자르기와 8비트 양자화만 거쳐 `--sr-dir/<데이터셋>/<image_id>.png`에 저장한다.

이어서 `scripts/run_pilot_diagnosis.py --sr-dir <sr-dir> --hr-dir <data-dir>/testsets --output-dir results/pilot/bicubic --edge-top-q <값>`
으로 파일럿 베이스라인(`results/pilot/`)과 분리된 폴더에 진단 4종을 남긴다.
`configs/fair_comparison.yaml`의 `pilot_determined.edge_top_q`가 확정되기 전에는 `--edge-top-q`로 값을 넘겨야 한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from sr_pipeline.engine.pilot import generate_bicubic_sr_results, resolve_testset_names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate x4 bicubic SR results from testset LR images for pilot diagnosis."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared data root (contains testsets/).")
    parser.add_argument(
        "--sr-dir",
        type=Path,
        required=True,
        help="Where to write bicubic SR results (<dataset>/<image_id>.png); must not exist yet.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Resize device."
    )
    parser.add_argument(
        "--fair-comparison-config",
        type=Path,
        default=Path("configs/fair_comparison.yaml"),
        help="Fair comparison conditions file (source of testset names).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with args.fair_comparison_config.open(encoding="utf-8") as file:
        dataset_names = resolve_testset_names(yaml.safe_load(file))
    testsets_dir = args.data_dir / "testsets"
    counts = generate_bicubic_sr_results(testsets_dir, args.sr_dir, dataset_names, device=args.device)
    print(f"sr_image_counts: {counts}")
    print(
        "next: scripts/run_pilot_diagnosis.py "
        "(--sr-dir: Bicubic SR 결과 폴더, --hr-dir: <data-dir>/testsets, --output-dir results/pilot/bicubic)"
    )


if __name__ == "__main__":
    main()
