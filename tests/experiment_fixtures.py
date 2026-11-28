"""T3.8과 T3.9에서 공용으로 쓰는 합성 데이터와 실행 CLI 픽스처."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from sr_pipeline.data.prepare import prepare_testsets
from sr_pipeline.engine import trainer

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
