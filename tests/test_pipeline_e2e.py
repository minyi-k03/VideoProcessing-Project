"""주 경계(구성 + 시드 -> 실행 기록) 종단간 CPU 테스트 (SPEC.md Testing Decisions §1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml

from experiment_fixtures import DATASETS, POOL_ID, REPO_ROOT, args_for, git, only_run_dir

pytestmark = pytest.mark.usefixtures("restore_global_random_state")

CONFIG_IDS = tuple(yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))["configs"])
METRICS = ["psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips"]
SUMMARY_COLUMNS = [
    "run_id", "commit", "config_id", "seed", "patch_pool_id", "switches", "dataset", *METRICS,
    "psnr_y_pre_rgb", "params", "infer_ms_net", "infer_ms_total", "train_minutes", "status",
]
PER_IMAGE_COLUMNS = ["run_id", "dataset", "image_id", *METRICS]


@pytest.mark.parametrize("config_id", CONFIG_IDS)
def test_all_configs_run_end_to_end_on_tiny_synthetic_data(
    config_id: str, cli: ModuleType, tmp_path: Path, fair_path: Path, prepared_data: Path, code_repo: Path
) -> None:
    seed = 0
    cli.main(args_for(tmp_path, fair_path, config=config_id, seed=seed, data_dir=prepared_data))

    run_dir = only_run_dir(tmp_path / "records")
    commit = git(code_repo, "rev-parse", "HEAD")
    assert run_dir.name == f"{commit}-{config_id}-s{seed}"

    summary = pd.read_csv(run_dir / "summary.csv", dtype={"commit": str})
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert sorted(summary["dataset"]) == sorted(DATASETS)
    assert set(summary["run_id"]) == {run_dir.name}
    assert set(summary["commit"]) == {commit}
    assert set(summary["config_id"]) == {config_id}
    assert set(summary["seed"]) == {seed}
    assert set(summary["patch_pool_id"]) == {POOL_ID}
    assert set(summary["status"]) == {"complete"}
    assert summary[METRICS].notna().all().all()

    per_image = pd.read_csv(run_dir / "per_image.csv")
    assert list(per_image.columns) == PER_IMAGE_COLUMNS
    assert set(per_image["run_id"]) == {run_dir.name}
    expected_images = {(dataset, image_id) for dataset in DATASETS for image_id in ("img0", "img1")}
    assert len(per_image) == len(expected_images)
    assert set(per_image[["dataset", "image_id"]].itertuples(index=False, name=None)) == expected_images
    assert per_image[METRICS].notna().all().all()

    if config_id != "bicubic":
        train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
        assert train_log["epochs_done"] == train_log["epochs_total"] == 1


def test_same_config_and_seed_reproducible(
    cli: ModuleType, tmp_path: Path, fair_path: Path, prepared_data: Path
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        cli.main(args_for(root, fair_path, config="AB", seed=0, data_dir=prepared_data))
    first_run, second_run = only_run_dir(first / "records"), only_run_dir(second / "records")

    timing = ["train_minutes", "infer_ms_net", "infer_ms_total"]
    first_summary = pd.read_csv(first_run / "summary.csv", float_precision="round_trip").drop(columns=timing)
    second_summary = pd.read_csv(second_run / "summary.csv", float_precision="round_trip").drop(columns=timing)
    pd.testing.assert_frame_equal(first_summary, second_summary, check_exact=True)

    first_per_image = pd.read_csv(first_run / "per_image.csv", float_precision="round_trip")
    second_per_image = pd.read_csv(second_run / "per_image.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(first_per_image, second_per_image, check_exact=True)
