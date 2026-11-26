"""파일럿 진단 두 개(예: 파일럿 베이스라인 vs Bicubic 하한)를 영상 단위로 비교하는 CLI (Task T3.1).

두 진단 출력 폴더(`run_pilot_diagnosis.py`의 `--output-dir`)를 받아 테스트셋별 비교 표
`<output-dir>/<name>.csv`와 기록 `<output-dir>/<name>.json`(맞춘 진단 설정, 두 진단의 커밋, 이 비교의 커밋)을 쓴다.
ADR-0004의 진단 근거는 이 표에서만 인용한다(ADR-0006).
`src/`·`scripts/`·`configs/`에 커밋되지 않은 변경이 있으면 멈춘다(`--allow-dirty`면 커밋 뒤에 `-dirty`를 붙여 기록).
입력 경로는 기록하지 않는다(Public 저장소).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_pipeline.diagnosis.compare import compare_diagnoses
from sr_pipeline.diagnosis.provenance import resolve_code_commit

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two pilot diagnosis outputs image by image.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Diagnosis output of the model (e.g. results/pilot).")
    parser.add_argument(
        "--reference-dir", type=Path, required=True, help="Diagnosis output of the reference (e.g. results/pilot/bicubic)."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write <name>.csv and <name>.json.")
    parser.add_argument("--name", default="baseline_vs_bicubic", help="Output file stem.")
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
        raise SystemExit(f"비교를 시작하지 않았습니다: {error}") from error

    table, settings = compare_diagnoses(args.model_dir, args.reference_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}.csv"
    json_path = args.output_dir / f"{args.name}.json"
    table.to_csv(csv_path, index=False)
    record = {**settings, "commit": commit}
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"outputs: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()
