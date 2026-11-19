"""1주차 파일럿 베이스라인 진단 4종 CLI 진입점 (SPEC.md / User Story 51, Task T2.5).

파일럿 SR 결과(`<sr-dir>/<데이터셋>/<image_id>.png`)와 HR 영상(`<hr-dir>/<데이터셋>/HR/<image_id>.png`)로
오차 지도, 에지·평탄 PSNR(Y), 주파수 스펙트럼, YCbCr 채널별 오차와 테스트셋별 최악 영상 목록을 만든다.

가장자리 잘라내기(`shave_border`)와 테스트셋 목록(`datasets`)은 공정 비교 조건 파일의 `fixed_rules`에서만
읽는다. HR 영상 폴더의 테스트셋이 그 목록과 다르면 멈춘다.
`edge_top_q`는 T4.2에서 확정되기 전이면 `--edge-top-q`로 넘겨야 하고, 확정 후에는 파일 값과 같아야 한다.
쓴 값과 출처, 이 저장소의 현재 커밋은 `<output-dir>/diagnosis_settings.json`에 남는다.
`src/`·`scripts/`·`configs/`에 커밋되지 않은 변경이 있으면 멈춘다(`--allow-dirty`면 커밋 뒤에 `-dirty`를 붙여 기록).

같은 `--output-dir`에 다시 실행하면 진단 산출물만 바뀐다(이전 오차 지도가 섞이지 않고, 실패하면 이전 결과 그대로).
파일럿 학습 설정 기록(`pilot_train_settings.json`) 등 진단이 만들지 않은 파일은 남는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sr_pipeline.diagnosis.diagnostics import run_pilot_diagnosis
from sr_pipeline.diagnosis.provenance import resolve_code_commit

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run even with uncommitted changes in src/scripts/configs (recorded commit gets a -dirty suffix).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        commit = resolve_code_commit(REPO_ROOT, allow_dirty=args.allow_dirty)
    except RuntimeError as error:
        raise SystemExit(f"진단을 시작하지 않았습니다: {error}") from error
    with args.fair_comparison_config.open(encoding="utf-8") as file:
        fair_comparison = yaml.safe_load(file)

    record = run_pilot_diagnosis(
        args.sr_dir,
        args.hr_dir,
        args.output_dir,
        fair_comparison,
        commit=commit,
        edge_top_q=args.edge_top_q,
    )
    for key, value in record.items():
        print(f"{key}: {value}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
