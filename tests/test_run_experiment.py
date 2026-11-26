"""`scripts/run_experiment.py` 바깥 동작 테스트 (T3.8, SPEC.md US2, 7, 31, 38–39).

구성 + 시드 + 경로 인자 → 실행 기록 폴더. 합성 패치 풀·테스트셋과 테스트 코드 안 점검용 fair 파일만 쓴다
(`pilot_determined`가 `null`인 실제 `configs/fair_comparison.yaml`로는 실행 조건을 확정할 수 없다, T4.2 전).
Colab식(`--data-archive`)과 Kaggle식(`--data-dir`에 사전 해제된 폴더)이 같은 결과를 내는지 본다.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from sr_pipeline.data.prepare import pack_archive, prepare_testsets
from sr_pipeline.engine import trainer
from sr_pipeline.engine.runtime_data import extract_data_archive, sha256_of

pytestmark = pytest.mark.usefixtures("restore_global_random_state")  # 학습이 set_model_seed를 부른다

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Set5", "Set14", "BSD100", "Urban100")
SCALE, PATCH_HR, POOL_SIZE, POOL_ID = 4, 16, 12, "synthetic_pool"
SYNTHETIC_PILOT_DETERMINED: dict[str, Any] = {
    "patch_pool_id": POOL_ID,
    "patch_pool_size": POOL_SIZE,
    "patch_size_hr": PATCH_HR,
    "patches_per_epoch": 8,
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "scheduler": "none",
    "augmentation": "rot90x4_hflip",
    "edge_loss_lambda": 0.1,
    "edge_top_q": 10,
    "representative_images": {name: {"median": "img0", "worst": "img1"} for name in DATASETS},
}


class StubLpips(torch.nn.Module):
    """AlexNet 가중치를 받지 않고, 한 변 64화소 미만 합성 영상도 받는 LPIPS 대역."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).pow(2).mean().view(1, 1, 1, 1)


@pytest.fixture(scope="module")
def prepared_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`data.prepare`가 만드는 것과 같은 구조(`patch_pool/`, `testsets/`)의 작은 합성 데이터 루트."""
    root = tmp_path_factory.mktemp("prepared")
    rng = np.random.default_rng(0)
    hr_root = root / "hr_src"
    for name in DATASETS:
        (hr_root / name).mkdir(parents=True)
        for index, (h, w) in enumerate(((32, 40), (28, 28))):
            base = np.linspace(0, 255, w, dtype=np.float64)[None, :, None].repeat(h, 0).repeat(3, 2)
            noisy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(hr_root / name / f"img{index}.png")
    data = root / "data"
    prepare_testsets(hr_root, data, SCALE, list(DATASETS), "bicubic", "matlab_imresize_antialias")
    pool = data / "patch_pool"
    pool.mkdir()
    np.save(pool / "hr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR, PATCH_HR, 3), dtype=np.uint8))
    np.save(pool / "lr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR // SCALE, PATCH_HR // SCALE, 3), dtype=np.uint8))
    (pool / "metadata.json").write_text(
        json.dumps({"patch_pool_id": POOL_ID, "scale": SCALE, "num_patches": POOL_SIZE}), encoding="utf-8"
    )
    return data


@pytest.fixture(scope="module")
def archive(prepared_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return pack_archive(prepared_data, tmp_path_factory.mktemp("drive") / "sr_data.zip")


@pytest.fixture
def fair_path(tmp_path: Path) -> Path:
    fair = yaml.safe_load((REPO_ROOT / "configs" / "fair_comparison.yaml").read_text(encoding="utf-8"))
    fair["pilot_determined"] = copy.deepcopy(SYNTHETIC_PILOT_DETERMINED)
    fair["infer_benchmark"].update(lr_height=12, lr_width=16, sr_height=48, sr_width=64, warmup_runs=1, repeat_runs=2)
    path = tmp_path / "fair_for_test.yaml"
    path.write_text(yaml.safe_dump(fair, allow_unicode=True), encoding="utf-8")
    return path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    """코드 커밋을 조회할 임시 저장소(`src/`·`scripts/`·`configs/`가 커밋된 상태)."""
    root = tmp_path / "code_repo"
    for relative in ("src/module.py", "scripts/run.py", "configs/rules.yaml"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def cli(code_repo: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_experiment", REPO_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", code_repo)
    real_evaluate = trainer.evaluate_model
    monkeypatch.setattr(
        trainer, "evaluate_model", lambda *args, **kwargs: real_evaluate(*args, lpips_model=StubLpips(), **kwargs)
    )
    return module


def args_for(tmp_path: Path, fair_path: Path, *, config: str = "baseline", seed: int = 0, **paths: Path) -> list[str]:
    """Colab·Kaggle에서 달라지는 것은 아래 경로 인자뿐이다."""
    argv = [
        "--config", config, "--seed", str(seed), "--device", "cpu",
        "--fair-config", str(fair_path),
        "--data-dir", str(paths.get("data_dir", tmp_path / "local_data")),
        "--output-dir", str(paths.get("output_dir", tmp_path / "records")),
        "--checkpoint-dir", str(paths.get("checkpoint_dir", tmp_path / "ckpt")),
    ]
    if "archive" in paths:
        argv += ["--data-archive", str(paths["archive"])]
    return argv


def only_run_dir(output_dir: Path) -> Path:
    (run_dir,) = [path for path in output_dir.iterdir() if path.is_dir()]
    return run_dir


def test_drive_style_archive_run_produces_run_record(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(args_for(tmp_path, fair_path, archive=archive))

    run_dir = only_run_dir(tmp_path / "records")
    assert {path.name for path in run_dir.iterdir()} == {"summary.csv", "per_image.csv", "train_log.json", "images"}
    summary = pd.read_csv(run_dir / "summary.csv")
    assert list(summary["dataset"]) == list(DATASETS)
    assert set(summary["config_id"]) == {"baseline"} and set(summary["seed"]) == {0}
    assert set(summary["status"]) == {"complete"} and set(summary["patch_pool_id"]) == {POOL_ID}
    assert len(pd.read_csv(run_dir / "per_image.csv")) == 2 * len(DATASETS)
    assert (tmp_path / "ckpt" / run_dir.name / "weights.pt").is_file()
    assert (tmp_path / "local_data" / "patch_pool" / "metadata.json").is_file()  # 런타임 로컬로 풀렸다

    output = capsys.readouterr()
    assert run_dir.name in output.out
    assert str(tmp_path) not in output.out + output.err  # 입력·저장 경로는 출력하지 않는다(Public 저장소)
    assert str(tmp_path) not in (run_dir / "train_log.json").read_text(encoding="utf-8")


def test_kaggle_style_prepared_input_folder_gives_the_same_record_and_is_not_modified(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    kaggle_input = tmp_path / "input"
    extract_data_archive(archive, kaggle_input / "datasets" / "someone" / "slug")  # 자동 해제돼 한 단계 감싸진 Dataset
    before = sorted(path.relative_to(kaggle_input) for path in kaggle_input.rglob("*"))

    cli.main(args_for(tmp_path, fair_path, data_dir=kaggle_input, output_dir=tmp_path / "kaggle_records"))

    assert sorted(path.relative_to(kaggle_input) for path in kaggle_input.rglob("*")) == before
    cli.main(args_for(tmp_path, fair_path, archive=archive, output_dir=tmp_path / "drive_records",
                      checkpoint_dir=tmp_path / "ckpt2"))
    kaggle_summary = pd.read_csv(only_run_dir(tmp_path / "kaggle_records") / "summary.csv")
    drive_summary = pd.read_csv(only_run_dir(tmp_path / "drive_records") / "summary.csv")
    # 같은 코드·구성·시드·데이터면 입력 경로와 상관없이 같은 학습 결과(시간 필드는 제외)다.
    timing = ["train_minutes", "infer_ms_net", "infer_ms_total"]
    pd.testing.assert_frame_equal(kaggle_summary.drop(columns=timing), drive_summary.drop(columns=timing))


def test_bicubic_runs_without_training_and_without_checkpoints(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    cli.main(args_for(tmp_path, fair_path, config="bicubic", archive=archive))

    run_dir = only_run_dir(tmp_path / "records")
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["config_id"]) == {"bicubic"} and set(summary["status"]) == {"complete"}
    assert (summary["params"] == 0).all() and (summary["train_minutes"] == 0).all()
    assert not (run_dir / "train_log.json").exists()
    assert not (tmp_path / "ckpt").exists() or not any((tmp_path / "ckpt").iterdir())


def test_same_run_is_not_overwritten(cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path) -> None:
    cli.main(args_for(tmp_path, fair_path, archive=archive))
    run_dir = only_run_dir(tmp_path / "records")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    with pytest.raises(FileExistsError):
        cli.main(args_for(tmp_path, fair_path, archive=archive))
    assert {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()} == before


def test_undetermined_fair_comparison_is_rejected_before_touching_data_or_outputs(
    cli: ModuleType, tmp_path: Path, archive: Path
) -> None:
    """실제 `configs/fair_comparison.yaml`은 T4.2 전이라 `null`이 남아 있다: 해제·학습 전에 멈춘다."""
    argv = args_for(tmp_path, REPO_ROOT / "configs" / "fair_comparison.yaml", archive=archive)

    with pytest.raises(ValueError, match="null"):
        cli.main(argv)
    for name in ("local_data", "records", "ckpt"):
        assert not (tmp_path / name).exists()


@pytest.mark.parametrize("override", [{"config": "not_a_config"}, {"seed": 7}])
def test_invalid_config_or_seed_is_rejected_before_touching_data(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, override: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        cli.main(args_for(tmp_path, fair_path, archive=archive, **override))
    assert not (tmp_path / "local_data").exists() and not (tmp_path / "records").exists()


def test_uncommitted_code_changes_are_rejected_before_unpacking_unless_allowed(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, code_repo: Path
) -> None:
    (code_repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="커밋되지 않은 변경"):
        cli.main(args_for(tmp_path, fair_path, archive=archive))
    assert not (tmp_path / "local_data").exists()

    with pytest.warns(UserWarning, match="커밋되지 않은 변경"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--allow-dirty"])
    assert only_run_dir(tmp_path / "records").name.startswith(git(code_repo, "rev-parse", "HEAD") + "-dirty-")


def test_wrong_archive_sha256_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    with pytest.raises(ValueError, match="sha256"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--archive-sha256", "0" * 64])
    assert not (tmp_path / "local_data").exists()

    cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--archive-sha256", sha256_of(archive)])
    assert only_run_dir(tmp_path / "records")


@pytest.mark.parametrize(
    ("config", "resume_run_id", "message"),
    [("baseline", "other-baseline-s0", "resume_run_id"), ("bicubic", "other-bicubic-s0", "학습하지 않는 구성")],
)
def test_invalid_resume_request_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, config: str, resume_run_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cli.main(args_for(tmp_path, fair_path, config=config, archive=archive) + ["--resume-run-id", resume_run_id])
    assert not (tmp_path / "local_data").exists() and not (tmp_path / "records").exists()


def test_dirty_run_cannot_be_resumed_and_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, code_repo: Path
) -> None:
    (code_repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    dirty_run_id = f"{git(code_repo, 'rev-parse', 'HEAD')}-dirty-baseline-s0"

    with pytest.warns(UserWarning), pytest.raises(ValueError, match="재개할 수 없습니다"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--allow-dirty", "--resume-run-id", dirty_run_id])
    assert not (tmp_path / "local_data").exists()


def test_default_config_paths_do_not_depend_on_the_working_directory(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--experiments-config`를 주지 않고 저장소 밖에서 실행해도 저장소의 `configs/experiments.yaml`을 쓴다."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cli.main(args_for(tmp_path, fair_path, archive=archive))

    assert only_run_dir(tmp_path / "records")
    defaults = cli.parse_args(["--config", "baseline", "--seed", "0", "--data-dir", "d", "--checkpoint-dir", "c"])
    assert defaults.output_dir == REPO_ROOT / "results" / "runs"
    assert defaults.fair_config == REPO_ROOT / "configs" / "fair_comparison.yaml"


def test_help_lists_the_path_arguments(cli: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--data-dir", "--data-archive", "--checkpoint-dir", "--output-dir", "--resume-run-id", "--allow-dirty"):
        assert option in help_text
