"""파일럿 베이스라인 학습·SR 출력 생성 바깥 동작 테스트 (SPEC.md US51, Task T2.5).

작은 합성 패치 풀·테스트셋만 쓰므로 로컬 CPU에서 돈다. 여기 쓰인 하이퍼파라미터 값은 테스트용이며,
`pilot_determined`가 `null`인 동안 코드에는 기본값을 두지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sr_pipeline.diagnosis.diagnostics import run_pilot_diagnosis
from sr_pipeline.engine.pilot import generate_sr_outputs, train_pilot_baseline

SCALE = 4
PATCH_HR = 16
POOL_SIZE = 12
TRAIN_KWARGS = dict(seed=0, epochs=2, patches_per_epoch=8, batch_size=4, learning_rate=1e-3, device="cpu")


def _write_pool(data_dir: Path, seed: int = 123) -> None:
    """`data.prepare.create_patch_pool`과 같은 파일 구조의 작은 합성 패치 풀을 만든다."""
    rng = np.random.default_rng(seed)
    pool_dir = data_dir / "patch_pool"
    pool_dir.mkdir(parents=True)
    hr = rng.integers(0, 256, (POOL_SIZE, PATCH_HR, PATCH_HR, 3), dtype=np.uint8)
    lr = rng.integers(0, 256, (POOL_SIZE, PATCH_HR // SCALE, PATCH_HR // SCALE, 3), dtype=np.uint8)
    np.save(pool_dir / "hr.npy", hr)
    np.save(pool_dir / "lr.npy", lr)
    metadata = {"patch_pool_id": "synthetic_pool", "scale": SCALE, "num_patches": POOL_SIZE}
    (pool_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    _write_pool(path)
    return path


def test_train_writes_checkpoint_and_finite_epoch_losses(data_dir: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt" / "pilot.pth"

    record = train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)

    assert checkpoint.is_file()
    assert record["patch_pool_id"] == "synthetic_pool"
    assert record["seed"] == 0 and record["epochs"] == 2
    assert len(record["epoch_losses"]) == 2
    assert all(math.isfinite(loss) for loss in record["epoch_losses"])
    assert record["train_minutes"] >= 0.0


def test_train_same_seed_reproduces_losses_and_different_seed_differs(data_dir: Path, tmp_path: Path) -> None:
    first = train_pilot_baseline(data_dir, tmp_path / "a.pth", **TRAIN_KWARGS)
    second = train_pilot_baseline(data_dir, tmp_path / "b.pth", **TRAIN_KWARGS)
    other = train_pilot_baseline(data_dir, tmp_path / "c.pth", **{**TRAIN_KWARGS, "seed": 1})

    assert first["epoch_losses"] == second["epoch_losses"]
    assert first["epoch_losses"] != other["epoch_losses"]


def test_train_requires_hyperparameters_without_defaults(data_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        train_pilot_baseline(data_dir, tmp_path / "x.pth", seed=0, device="cpu")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "override",
    [{"epochs": 0}, {"batch_size": 0}, {"patches_per_epoch": 0}, {"learning_rate": 0.0}, {"learning_rate": float("nan")}],
)
def test_train_rejects_invalid_hyperparameters_before_writing(
    data_dir: Path, tmp_path: Path, override: dict
) -> None:
    checkpoint = tmp_path / "bad.pth"

    with pytest.raises(ValueError):
        train_pilot_baseline(data_dir, checkpoint, **{**TRAIN_KWARGS, **override})

    assert not checkpoint.exists()


def _write_testsets(data_dir: Path, names: tuple[str, ...], sizes: tuple[tuple[int, int], ...]) -> None:
    """`data.prepare.prepare_testsets`와 같은 구조(`testsets/<데이터셋>/{HR,LR}/<image_id>.png`)의 합성 테스트셋."""
    rng = np.random.default_rng(7)
    for name in names:
        for index, (height, width) in enumerate(sizes):
            for kind, divisor in (("HR", 1), ("LR", SCALE)):
                folder = data_dir / "testsets" / name / kind
                folder.mkdir(parents=True, exist_ok=True)
                array = rng.integers(0, 256, (height // divisor, width // divisor, 3), dtype=np.uint8)
                Image.fromarray(array).save(folder / f"img{index}.png")


def test_generate_sr_outputs_writes_hr_sized_png_per_test_image(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA", "SetB"), ((32, 48), (24, 24)))
    checkpoint = tmp_path / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    sr_dir = tmp_path / "sr"

    generate_sr_outputs(checkpoint, data_dir / "testsets", sr_dir, ["SetA", "SetB"], device="cpu")

    for name in ("SetA", "SetB"):
        for image_id in ("img0", "img1"):
            with Image.open(sr_dir / name / f"{image_id}.png") as sr, Image.open(
                data_dir / "testsets" / name / "HR" / f"{image_id}.png"
            ) as hr:
                assert sr.mode == "RGB"
                assert sr.size == hr.size


def test_generate_sr_outputs_rejects_missing_dataset_before_writing(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    checkpoint = tmp_path / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    sr_dir = tmp_path / "sr"

    with pytest.raises(ValueError):
        generate_sr_outputs(checkpoint, data_dir / "testsets", sr_dir, ["SetA", "Missing"], device="cpu")

    assert not sr_dir.exists()


def test_train_to_pilot_diagnosis_end_to_end(data_dir: Path, tmp_path: Path) -> None:
    """학습 → SR 출력 → 진단 4종이 이어져 `results/pilot/`에 들어갈 산출물이 만들어진다."""
    _write_testsets(data_dir, ("SetA",), ((32, 32), (40, 24)))
    checkpoint, sr_dir, output_dir = tmp_path / "pilot.pth", tmp_path / "sr", tmp_path / "pilot_out"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    generate_sr_outputs(checkpoint, data_dir / "testsets", sr_dir, ["SetA"], device="cpu")
    fair_comparison = {"fixed_rules": {"shave_border": 4}, "pilot_determined": {"edge_top_q": None}}

    settings = run_pilot_diagnosis(
        sr_dir, data_dir / "testsets", output_dir, fair_comparison, edge_top_q=20.0
    )

    assert settings["shave_border"] == 4
    for name in ("per_image_diagnosis.csv", "summary_diagnosis.csv", "worst_images.csv", "diagnosis_settings.json"):
        assert (output_dir / name).is_file()


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pilot_baseline.py"
    spec = importlib.util.spec_from_file_location("train_pilot_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(data_dir: Path, tmp_path: Path, config: Path) -> list[str]:
    return [
        "--data-dir", str(data_dir),
        "--checkpoint-path", str(tmp_path / "ckpt" / "pilot.pth"),
        "--sr-dir", str(tmp_path / "sr"),
        "--output-dir", str(tmp_path / "out"),
        "--seed", "0", "--epochs", "1", "--patches-per-epoch", "8", "--batch-size", "4",
        "--learning-rate", "0.001", "--device", "cpu",
        "--fair-comparison-config", str(config),
    ]


def test_cli_trains_writes_sr_outputs_and_settings_without_paths(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((32, 32),))
    config = tmp_path / "fair.yaml"
    config.write_text("fixed_rules:\n  datasets: [SetA]\n", encoding="utf-8")

    _load_cli().main(_cli_args(data_dir, tmp_path, config))

    assert (tmp_path / "sr" / "SetA" / "img0.png").is_file()
    settings_text = (tmp_path / "out" / "pilot_train_settings.json").read_text(encoding="utf-8")
    settings = json.loads(settings_text)
    assert settings["patch_pool_id"] == "synthetic_pool" and settings["sr_image_counts"] == {"SetA": 1}
    assert str(tmp_path) not in settings_text  # Public 저장소에 커밋되므로 경로를 남기지 않는다.


def test_cli_requires_hyperparameters(data_dir: Path, tmp_path: Path) -> None:
    args = _cli_args(data_dir, tmp_path, tmp_path / "fair.yaml")
    args.remove("--epochs")
    args.remove("1")

    with pytest.raises(SystemExit):
        _load_cli().main(args)
