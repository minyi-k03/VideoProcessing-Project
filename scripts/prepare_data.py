"""DIV2K 패치 풀 및 테스트셋 LR 생성 CLI 진입점 (SPEC.md / ADR-0001).

배율·열화 모델·bicubic 규칙·테스트셋 이름은 공정 비교 조건 파일(configs/fair_comparison.yaml)의
fixed_rules에서 읽는다. 패치 크기·개수는 파일럿 후 확정되는 값이라 코드에 기본값을 두지 않고 인자로 받는다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from sr_pipeline.data.prepare import create_patch_pool, pack_archive, prepare_testsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared patch pool and x4 testset LR images.")
    parser.add_argument("--div2k-hr-dir", type=Path, required=True, help="DIV2K train HR directory.")
    parser.add_argument("--testsets-hr-dir", type=Path, required=True, help="Testsets HR root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output data directory.")
    parser.add_argument(
        "--archive-path",
        type=Path,
        required=True,
        help="Output zip file (must be outside --output-dir).",
    )
    parser.add_argument("--patch-size-hr", type=int, required=True, help="HR patch size.")
    parser.add_argument("--num-patches", type=int, required=True, help="Total patches in shared pool.")
    parser.add_argument("--pool-seed", type=int, required=True, help="Deterministic seed for patch pool creation.")
    parser.add_argument(
        "--fair-comparison-config",
        type=Path,
        default=Path("configs/fair_comparison.yaml"),
        help="Fair comparison conditions file (source of scale, degradation, bicubic rule, testset names).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.fair_comparison_config.open(encoding="utf-8") as file:
        rules = yaml.safe_load(file)["fixed_rules"]

    # 짧은 테스트셋 단계를 먼저 돌려, 테스트셋 경로·구성 오류가 긴 패치 풀 생성이 끝난 뒤에야 드러나지 않게 한다.
    prepare_testsets(
        args.testsets_hr_dir,
        args.output_dir,
        scale=rules["scale"],
        dataset_names=rules["datasets"],
        degradation=rules["degradation"],
        bicubic_rule=rules["bicubic_rule"],
    )
    metadata = create_patch_pool(
        args.div2k_hr_dir,
        args.output_dir,
        scale=rules["scale"],
        patch_size_hr=args.patch_size_hr,
        num_patches=args.num_patches,
        pool_seed=args.pool_seed,
        degradation=rules["degradation"],
        bicubic_rule=rules["bicubic_rule"],
    )
    archive_path = pack_archive(args.output_dir, args.archive_path)

    for key, value in asdict(metadata).items():
        print(f"{key}: {value}")
    print(f"archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
