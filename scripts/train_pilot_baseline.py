"""파일럿 베이스라인 1시드 학습 + SR 출력 생성 CLI 진입점 (SPEC.md / User Story 51, Task T2.5).

정식 파이프라인(T3.7)이 끝나기 전에 진단 입력을 만드는 파일럿 전용 스크립트다. 순서:
1. `<data-dir>/patch_pool`로 베이스라인 ESPCN을 학습하고 체크포인트를 `--checkpoint-path`에 저장한다.
2. `<data-dir>/testsets/<데이터셋>/LR`의 SR 결과를 `--sr-dir`에 만든다. 데이터셋 이름은 공정 비교 조건 파일의
   `fixed_rules.datasets`에서 읽는다.
3. 쓴 학습 설정을 `--output-dir/pilot_train_settings.json`에 남긴다(입력·출력 경로는 기록하지 않는다).
이어서 `scripts/run_pilot_diagnosis.py --sr-dir <sr-dir> --hr-dir <data-dir>/testsets`로 진단 4종을 돌린다.

에폭·에폭당 패치·배치·학습률은 `pilot_determined`가 `null`인 동안 코드에 기본값을 두지 않고 모두 필수 인자다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from sr_pipeline.engine.pilot import generate_sr_outputs, train_pilot_baseline

SETTINGS_FILENAME = "pilot_train_settings.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pilot baseline (1 seed) and write SR outputs for diagnosis.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared data root (patch_pool/, testsets/).")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Checkpoint file to write (keep on Drive).")
    parser.add_argument("--sr-dir", type=Path, required=True, help="Where to write SR outputs (<dataset>/<image_id>.png).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/pilot"), help="Where to write pilot_train_settings.json."
    )
    parser.add_argument("--seed", type=int, required=True, help="Pilot seed (one seed only).")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patches-per-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training/inference device."
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
        dataset_names = yaml.safe_load(file)["fixed_rules"]["datasets"]

    record = train_pilot_baseline(
        args.data_dir,
        args.checkpoint_path,
        seed=args.seed,
        epochs=args.epochs,
        patches_per_epoch=args.patches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    counts = generate_sr_outputs(
        args.checkpoint_path, args.data_dir / "testsets", args.sr_dir, dataset_names, device=args.device
    )
    record["sr_image_counts"] = counts

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / SETTINGS_FILENAME).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in record.items():
        print(f"{key}: {value}")
    print(f"next: python scripts/run_pilot_diagnosis.py --sr-dir {args.sr_dir} --hr-dir {args.data_dir / 'testsets'}")


if __name__ == "__main__":
    main()
