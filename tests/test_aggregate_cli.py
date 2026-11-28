"""`scripts/aggregate_results.py` CLI 및 `main` 병합 실행 기록 검증 테스트 (T6.3, SPEC.md US58, ADR-0006)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없는 환경")

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
SUMMARY_COLUMNS = [
    "run_id", "commit", "config_id", "seed", "patch_pool_id", "switches", "dataset",
    "psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips", "psnr_y_pre_rgb",
    "params", "infer_ms_net", "infer_ms_total", "train_minutes", "status",
]  # fmt: skip
PER_IMAGE_COLUMNS = ["run_id", "dataset", "image_id", "psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _write_run(runs_dir: Path, config_id: str, seed: int, psnr_y: float) -> None:
    run_id = f"c0-{config_id}-s{seed}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "run_id": run_id,
        "commit": "c0",
        "config_id": config_id,
        "seed": seed,
        "patch_pool_id": "pool_x",
        "switches": "none" if config_id in ("baseline", "bicubic") else config_id,
        "dataset": "BSD100",
        "psnr_y": psnr_y,
        "ssim_y": 0.9,
        "psnr_rgb": 28.0,
        "edge_psnr": 25.0,
        "lpips": 0.1,
        "psnr_y_pre_rgb": None,
        "params": 0 if config_id == "bicubic" else 37200,
        "infer_ms_net": 0.0 if config_id == "bicubic" else 1.5,
        "infer_ms_total": 0.6 if config_id == "bicubic" else 1.6,
        "train_minutes": 0.0 if config_id == "bicubic" else 10.0,
        "status": "complete",
    }
    pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS).to_csv(run_dir / "summary.csv", index=False)
    per_image_rows = [
        {
            "run_id": run_id,
            "dataset": "BSD100",
            "image_id": img_id,
            "psnr_y": psnr_y + delta,
            "ssim_y": 0.9,
            "psnr_rgb": 28.0,
            "edge_psnr": 25.0,
            "lpips": 0.1,
        }
        for img_id, delta in (("b0", 0.2), ("b1", -0.2))
    ]
    pd.DataFrame(per_image_rows, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
    if config_id != "bicubic":
        (run_dir / "train_log.json").write_text(
            json.dumps({"run_id": run_id, "config_id": config_id, "seed": seed, "epochs_done": 1, "epochs_total": 1, "epoch_losses": [0.1]}),
            encoding="utf-8",
        )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "results" / "runs").mkdir(parents=True)
    (root / "results" / "aggregated").mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "configs" / "fair_comparison.yaml", root / "configs" / "fair_comparison.yaml")
    shutil.copyfile(REPO_ROOT / "configs" / "experiments.yaml", root / "configs" / "experiments.yaml")
    for s in SEEDS:
        _write_run(root / "results" / "runs", "baseline", s, 30.0)
    (root / "results" / "aggregated" / ".gitkeep").write_text("", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init main")
    return root


@pytest.fixture
def cli(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("aggregate_results", REPO_ROOT / "scripts" / "aggregate_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", git_repo)
    monkeypatch.setattr(module, "DEFAULT_RUNS_DIR", git_repo / "results" / "runs")
    monkeypatch.setattr(module, "DEFAULT_OUTPUT_DIR", git_repo / "results" / "aggregated")
    monkeypatch.setattr(module, "DEFAULT_FAIR_CONFIG", git_repo / "configs" / "fair_comparison.yaml")
    monkeypatch.setattr(module, "DEFAULT_EXPERIMENTS_CONFIG", git_repo / "configs" / "experiments.yaml")
    return module


def test_rejects_unmerged_run_on_feature_branch(git_repo: Path, cli: ModuleType) -> None:
    _git(git_repo, "checkout", "-q", "-b", "YongMin")
    for s in SEEDS:
        _write_run(git_repo / "results" / "runs", "A", s, 30.5)
    _git(git_repo, "add", "results/runs")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "unmerged A")

    with pytest.raises(RuntimeError, match="main"):
        cli.main([])
    assert not (git_repo / "results" / "aggregated" / "tables").exists()


def test_rejects_untracked_or_uncommitted_run_in_runs_dir(git_repo: Path, cli: ModuleType) -> None:
    _write_run(git_repo / "results" / "runs", "A", 0, 30.5)

    with pytest.raises(RuntimeError, match="커밋"):
        cli.main([])


def test_rejects_stale_branch_missing_run_merged_in_main(git_repo: Path, cli: ModuleType) -> None:
    _git(git_repo, "checkout", "-q", "-b", "YongMin")
    _git(git_repo, "checkout", "-q", "main")
    for s in SEEDS:
        _write_run(git_repo / "results" / "runs", "A", s, 30.5)
    _git(git_repo, "add", "results/runs")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "merged A on main")
    _git(git_repo, "checkout", "-q", "YongMin")

    with pytest.raises(RuntimeError, match="main"):
        cli.main([])


def test_allows_aggregation_when_runs_dir_matches_main(git_repo: Path, cli: ModuleType) -> None:
    _git(git_repo, "checkout", "-q", "-b", "YongMin")
    for s in SEEDS:
        _write_run(git_repo / "results" / "runs", "A", s, 30.5)
    _git(git_repo, "add", "results/runs")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "add A")
    _git(git_repo, "checkout", "-q", "main")
    _git(git_repo, "merge", "-q", "--no-ff", "YongMin", "-m", "merge YongMin")

    cli.main([])

    out_dir = git_repo / "results" / "aggregated"
    assert (out_dir / "tables" / "results.csv").is_file()
    assert (out_dir / "tables" / "ablation.md").is_file()
    assert (out_dir / "figures" / "per_image_delta_distribution_BSD100.png").is_file()
    assert (out_dir / "manifest.json").is_file()


def test_allow_unmerged_protects_default_output_dir_and_allows_preview_dir(
    git_repo: Path, cli: ModuleType, tmp_path: Path
) -> None:
    _git(git_repo, "checkout", "-q", "-b", "YongMin")
    for s in SEEDS:
        _write_run(git_repo / "results" / "runs", "A", s, 30.5)
    _git(git_repo, "add", "results/runs")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "unmerged A")

    with pytest.raises(ValueError, match="results/aggregated"):
        cli.main(["--allow-unmerged"])

    preview_dir = tmp_path / "preview"
    cli.main(["--allow-unmerged", "--output-dir", str(preview_dir)])
    assert (preview_dir / "tables" / "results.csv").is_file()
    assert not (git_repo / "results" / "aggregated" / "tables").exists()


def test_non_default_configs_refuse_default_output_dir(git_repo: Path, cli: ModuleType, tmp_path: Path) -> None:
    custom_fair = tmp_path / "fair.yaml"
    shutil.copyfile(git_repo / "configs" / "fair_comparison.yaml", custom_fair)
    with pytest.raises(ValueError, match="results/aggregated"):
        cli.main(["--fair-config", str(custom_fair)])


def test_non_main_ref_refuses_default_output_dir(git_repo: Path, cli: ModuleType) -> None:
    with pytest.raises(ValueError, match="results/aggregated"):
        cli.main(["--main-ref", "HEAD"])


def test_rejects_git_info_exclude_unmerged_run_in_runs_dir(git_repo: Path, cli: ModuleType) -> None:
    _write_run(git_repo / "results" / "runs", "A", 0, 35.0)
    exclude_file = git_repo / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    exclude_file.write_text("results/runs/c0-A-s0/\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="main"):
        cli.main([])
    assert not (git_repo / "results" / "aggregated" / "tables").exists()


def test_rejects_locally_modified_fair_config(git_repo: Path, cli: ModuleType) -> None:
    fair_path = git_repo / "configs" / "fair_comparison.yaml"
    fair_path.write_text(fair_path.read_text(encoding="utf-8").replace("max_hours: 3.0", "max_hours: 9.0"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="설정 파일"):
        cli.main([])
    assert not (git_repo / "results" / "aggregated" / "tables").exists()


def test_rejects_unmerged_fair_config_on_feature_branch(git_repo: Path, cli: ModuleType) -> None:
    _git(git_repo, "checkout", "-q", "-b", "YongMin")
    fair_path = git_repo / "configs" / "fair_comparison.yaml"
    fair_path.write_text(fair_path.read_text(encoding="utf-8").replace("max_hours: 3.0", "max_hours: 9.0"), encoding="utf-8")
    _git(git_repo, "add", "configs/fair_comparison.yaml")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "edit max_hours")

    with pytest.raises(RuntimeError, match="main"):
        cli.main([])
    assert not (git_repo / "results" / "aggregated" / "tables").exists()


def test_load_run_records_restricts_to_allowed_run_ids(tmp_path: Path) -> None:
    from sr_pipeline.reporting.aggregate import load_run_records

    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "baseline", 0, 30.0)
    _write_run(runs_dir, "A", 0, 31.0)
    summary, per_image = load_run_records(runs_dir, allowed_run_ids={"c0-baseline-s0"})
    assert set(summary["run_id"]) == {"c0-baseline-s0"}
    assert set(per_image["run_id"]) == {"c0-baseline-s0"}


def test_rejects_skip_worktree_modified_csv_and_config(git_repo: Path, cli: ModuleType) -> None:
    csv_path = git_repo / "results" / "runs" / "c0-baseline-s0" / "summary.csv"
    _git(git_repo, "update-index", "--skip-worktree", "results/runs/c0-baseline-s0/summary.csv")
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace(",30.0,", ",45.0,"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="skip-worktree"):
        cli.main([])
    assert not (git_repo / "results" / "aggregated" / "tables").exists()

    # CSV를 복구하고 fair_comparison.yaml에만 skip-worktree 변조를 걸어도 거부되어야 한다.
    _git(git_repo, "update-index", "--no-skip-worktree", "results/runs/c0-baseline-s0/summary.csv")
    _git(git_repo, "checkout", "--", "results/runs/c0-baseline-s0/summary.csv")
    fair_path = git_repo / "configs" / "fair_comparison.yaml"
    _git(git_repo, "update-index", "--skip-worktree", "configs/fair_comparison.yaml")
    fair_path.write_text(fair_path.read_text(encoding="utf-8").replace("max_hours: 3.0", "max_hours: 99.0"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="skip-worktree"):
        cli.main([])


def test_rejects_symlink_committed_in_main(git_repo: Path, cli: ModuleType) -> None:
    # OS symlink 권한 없이도 git index의 120000(symlink) 모드를 재현해 거부되는지 검증한다.
    target_blob = subprocess.run(
        ["git", "-C", str(git_repo), "hash-object", "-w", "--stdin"],
        input="../c0-baseline-s0/summary.csv",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _git(
        git_repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"120000,{target_blob},results/runs/c0-link-s0/summary.csv",
    )
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "commit symlink")
    _git(git_repo, "checkout", "--", "results/runs/c0-link-s0/summary.csv")

    with pytest.raises(RuntimeError, match="일반 파일"):
        cli.main([])


def test_allows_korean_image_filenames_merged_in_main(git_repo: Path, cli: ModuleType) -> None:
    import numpy as np
    from PIL import Image

    img_dir = git_repo / "results" / "runs" / "c0-baseline-s0" / "images" / "Set14"
    img_dir.mkdir(parents=True, exist_ok=True)
    panel = np.full((16, 48, 3), 120, dtype=np.uint8)
    Image.fromarray(panel).save(img_dir / "나비_median.png")
    Image.fromarray(panel).save(img_dir / "나비_median_roi.png")
    _git(git_repo, "config", "core.quotePath", "true")
    _git(git_repo, "add", "results/runs")
    _git(git_repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "add korean image")

    cli.main([])

    assert (git_repo / "results" / "aggregated" / "figures" / "comparison_Set14_나비_median.png").is_file()
