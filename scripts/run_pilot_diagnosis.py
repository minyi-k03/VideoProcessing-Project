"""1주차 파일럿 베이스라인 진단 4종 CLI 진입점 (SPEC.md / User Story 51, Task T2.5).

파일럿 SR 결과(`<sr-dir>/<데이터셋>/<image_id>.png`)와 HR 영상(`<hr-dir>/<데이터셋>/HR/<image_id>.png`)로
오차 지도, 에지·평탄 PSNR(Y), 주파수 스펙트럼, YCbCr 채널별 오차와 테스트셋별 최악 영상 목록을 만든다.

가장자리 잘라내기(`shave_border`)는 공정 비교 조건 파일의 `fixed_rules`에서만 읽는다.
`edge_top_q`는 T4.2에서 확정되기 전이면 `--edge-top-q`로 넘겨야 하고, 확정 후에는 파일 값과 같아야 한다.
쓴 값과 출처는 `<output-dir>/diagnosis_settings.json`에 남는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sr_pipeline.diagnosis.diagnostics import run_pilot_diagnosis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot baseline diagnosis suite (error map, edge/flat, FFT, YCbCr).")
    parser.add_argument("--sr-dir", type=Path, required=True, help="Pilot SR outputs root (<dataset>/<image_id>.png).")
    parser.add_argument("--hr-dir", type=Path, required=True, help="Testsets root (<dataset>/HR/<image_id>.png).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot"), help="Pilot diagnosis output dir.")
    parser.add_argument(
        "--edge-top-q",
        type=float,
        default=None,
        help="Top q%% Sobel edge region. Required while pilot_determined.edge_top_q is null; "
        "must match it once fixed.",
    )
    parser.add_argument(
        "--fair-comparison-config",
        type=Path,
        default=Path("configs/fair_comparison.yaml"),
        help="Fair comparison conditions file (source of shave_border and edge_top_q).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with args.fair_comparison_config.open(encoding="utf-8") as file:
        fair_comparison = yaml.safe_load(file)

    record = run_pilot_diagnosis(
        args.sr_dir,
        args.hr_dir,
        args.output_dir,
        fair_comparison,
        edge_top_q=args.edge_top_q,
    )
    for key, value in record.items():
        print(f"{key}: {value}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
