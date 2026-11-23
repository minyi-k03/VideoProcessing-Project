"""파일럿 베이스라인 학습·SR 결과 생성 바깥 동작 테스트 (SPEC.md US51, Task T2.5).

작은 합성 패치 풀·테스트셋만 쓰므로 로컬 CPU에서 돈다. 여기 쓰인 학습 설정 값은 테스트용이며,
`pilot_determined`가 `null`인 동안 코드에는 기본값을 두지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from sr_pipeline.common.seed import create_data_rng
from sr_pipeline.diagnosis.diagnostics import run_pilot_diagnosis
from sr_pipeline.engine.pilot import (
    collect_testset_lr_images,
    generate_sr_results,
    sample_epoch_batches,
    train_pilot_baseline,
)
from sr_pipeline.models.espcn import ESPCN

SCALE = 4
PATCH_HR = 16
POOL_SIZE = 12
TRAIN_KWARGS = dict(
    seed=0,
    epochs=2,
    patches_per_epoch=8,
    batch_size=4,
    learning_rate=1e-3,
    optimizer="adam",
    scheduler="none",
    augmentation="rot90x4_hflip",
    device="cpu",
)


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


def _upscale_by_repeat(lr: np.ndarray) -> np.ndarray:
    """(…, h, w, 3) → (…, h·4, w·4, 3) 화소 복제. 회전·뒤집기와 교환되므로 HR·LR 대응의 독립 기준이 된다."""
    return np.repeat(np.repeat(lr, SCALE, axis=-3), SCALE, axis=-2)


def test_epoch_batches_keep_hr_lr_spatial_correspondence_under_augmentation() -> None:
    rng = np.random.default_rng(1)
    lr_pool = rng.integers(0, 256, (6, 4, 4, 3), dtype=np.uint8)
    hr_pool = _upscale_by_repeat(lr_pool)

    batches = list(
        sample_epoch_batches(
            hr_pool, lr_pool, create_data_rng(0), patches_per_epoch=40, batch_size=8, augmentation="rot90x4_hflip"
        )
    )

    for lr, hr in batches:
        np.testing.assert_array_equal(hr, _upscale_by_repeat(lr))
    originals = {patch.tobytes() for patch in lr_pool}
    augmented = [patch for lr, _ in batches for patch in lr if patch.tobytes() not in originals]
    assert augmented, "증강(회전·뒤집기)이 한 번도 적용되지 않았다"


def test_epoch_batches_without_augmentation_keeps_original_patches() -> None:
    rng = np.random.default_rng(1)
    lr_pool = rng.integers(0, 256, (6, 4, 4, 3), dtype=np.uint8)
    hr_pool = _upscale_by_repeat(lr_pool)

    batches = list(
        sample_epoch_batches(
            hr_pool, lr_pool, create_data_rng(0), patches_per_epoch=12, batch_size=4, augmentation="none"
        )
    )

    originals = {patch.tobytes() for patch in lr_pool}
    for lr, hr in batches:
        np.testing.assert_array_equal(hr, _upscale_by_repeat(lr))
        assert all(patch.tobytes() in originals for patch in lr)


def test_epoch_batches_cover_pool_evenly_when_epoch_exceeds_pool() -> None:
    # 패치마다 단색(값 = 인덱스 × 10)이라 증강 후에도 어떤 패치인지 알 수 있다.
    lr_pool = np.stack([np.full((4, 4, 3), index * 10, dtype=np.uint8) for index in range(5)])
    hr_pool = _upscale_by_repeat(lr_pool)

    batches = list(
        sample_epoch_batches(
            hr_pool, lr_pool, create_data_rng(0), patches_per_epoch=13, batch_size=4, augmentation="rot90x4_hflip"
        )
    )

    assert [len(lr) for lr, _ in batches] == [4, 4, 4, 1]
    seen = np.concatenate([lr[:, 0, 0, 0] // 10 for lr, _ in batches])
    counts = np.bincount(seen, minlength=5)
    assert counts.sum() == 13
    assert set(counts.tolist()) <= {2, 3}  # 풀 2바퀴(10개) + 세 번째 바퀴 3개


def test_train_writes_checkpoint_and_finite_epoch_losses(data_dir: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt" / "pilot.pth"

    record = train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)

    assert checkpoint.is_file()
    assert record["patch_pool_id"] == "synthetic_pool"
    assert record["seed"] == 0 and record["epochs"] == 2
    assert len(record["epoch_losses"]) == 2
    assert all(math.isfinite(loss) for loss in record["epoch_losses"])
    assert record["train_minutes"] >= 0.0
    # 호출자가 넘긴 학습 조건도 기록에 남아야 진단 근거를 재현할 수 있다.
    assert (record["optimizer"], record["scheduler"], record["augmentation"]) == ("adam", "none", "rot90x4_hflip")


def test_train_same_seed_reproduces_losses_and_different_seed_differs(data_dir: Path, tmp_path: Path) -> None:
    first = train_pilot_baseline(data_dir, tmp_path / "a.pth", **TRAIN_KWARGS)
    second = train_pilot_baseline(data_dir, tmp_path / "b.pth", **TRAIN_KWARGS)
    other = train_pilot_baseline(data_dir, tmp_path / "c.pth", **{**TRAIN_KWARGS, "seed": 1})

    assert first["epoch_losses"] == second["epoch_losses"]
    assert first["epoch_losses"] != other["epoch_losses"]
    weights_a = torch.load(tmp_path / "a.pth", weights_only=True)
    weights_b = torch.load(tmp_path / "b.pth", weights_only=True)
    assert all(torch.equal(weights_a[key], weights_b[key]) for key in weights_a)


@pytest.mark.parametrize(
    "omitted",
    ["epochs", "patches_per_epoch", "batch_size", "learning_rate", "optimizer", "scheduler", "augmentation"],
)
def test_train_requires_training_settings_without_defaults(data_dir: Path, tmp_path: Path, omitted: str) -> None:
    kwargs = {key: value for key, value in TRAIN_KWARGS.items() if key != omitted}
    with pytest.raises(TypeError):
        train_pilot_baseline(data_dir, tmp_path / "x.pth", **kwargs)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 0},
        {"epochs": True},
        {"batch_size": 0},
        {"patches_per_epoch": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"optimizer": "sgd"},
        {"scheduler": "cosine"},
        {"augmentation": "unknown"},
        {"seed": -1},
    ],
)
def test_train_rejects_invalid_training_settings_before_writing(
    data_dir: Path, tmp_path: Path, override: dict
) -> None:
    checkpoint = tmp_path / "bad.pth"

    with pytest.raises(ValueError):
        train_pilot_baseline(data_dir, checkpoint, **{**TRAIN_KWARGS, **override})

    assert not checkpoint.exists()


@pytest.mark.parametrize("pool_shapes", [((0, 16, 16, 3), (0, 4, 4, 3)), ((2, 16, 16, 3), (2, 8, 8, 3))])
def test_train_rejects_empty_or_mismatched_pool_before_writing(tmp_path: Path, pool_shapes: tuple) -> None:
    pool_dir = tmp_path / "data" / "patch_pool"
    pool_dir.mkdir(parents=True)
    np.save(pool_dir / "hr.npy", np.zeros(pool_shapes[0], dtype=np.uint8))
    np.save(pool_dir / "lr.npy", np.zeros(pool_shapes[1], dtype=np.uint8))
    (pool_dir / "metadata.json").write_text(json.dumps({"patch_pool_id": "bad"}), encoding="utf-8")
    checkpoint = tmp_path / "bad.pth"

    with pytest.raises(ValueError):
        train_pilot_baseline(tmp_path / "data", checkpoint, **TRAIN_KWARGS)

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


def test_generate_sr_results_writes_hr_sized_png_per_test_image(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA", "SetB"), ((32, 48), (24, 24)))
    checkpoint = tmp_path / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    sr_dir = tmp_path / "sr"

    counts = generate_sr_results(checkpoint, data_dir / "testsets", sr_dir, ["SetA", "SetB"], device="cpu")

    assert counts == {"SetA": 2, "SetB": 2}
    for name in ("SetA", "SetB"):
        for image_id in ("img0", "img1"):
            with Image.open(sr_dir / name / f"{image_id}.png") as sr, Image.open(
                data_dir / "testsets" / name / "HR" / f"{image_id}.png"
            ) as hr:
                assert sr.mode == "RGB"
                assert sr.size == hr.size


def test_generate_sr_results_rejects_missing_dataset_before_writing(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    checkpoint = tmp_path / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    sr_dir = tmp_path / "sr"

    with pytest.raises(ValueError):
        generate_sr_results(checkpoint, data_dir / "testsets", sr_dir, ["SetA", "Missing"], device="cpu")

    assert not sr_dir.exists()


@pytest.mark.parametrize("bias, expected", [(0.5, 128), (1.7, 255), (-0.2, 0)])
def test_generate_sr_results_clamps_and_quantizes_to_8bit(
    data_dir: Path, tmp_path: Path, bias: float, expected: int
) -> None:
    """마지막 층 가중치 0·편향 c면 자르기 전 SR 결과가 모든 화소에서 c다. 저장값은 round(clamp(c)·255)여야 한다."""
    _write_testsets(data_dir, ("SetA",), ((16, 24),))
    model = ESPCN()
    with torch.no_grad():
        model.last.weight.zero_()
        model.last.bias.fill_(bias)
    checkpoint = tmp_path / "constant.pth"
    torch.save(model.state_dict(), checkpoint)

    generate_sr_results(checkpoint, data_dir / "testsets", tmp_path / "sr", ["SetA"], device="cpu")

    with Image.open(tmp_path / "sr" / "SetA" / "img0.png") as image:
        pixels = np.asarray(image)
    assert pixels.shape == (16, 24, 3)
    assert (pixels == expected).all()


def test_collect_testset_lr_images_rejects_empty_dataset_names(data_dir: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))

    with pytest.raises(ValueError):
        collect_testset_lr_images(data_dir / "testsets", [])


def test_generate_sr_results_refuses_existing_sr_dir(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    checkpoint = tmp_path / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    sr_dir = tmp_path / "sr"
    (sr_dir / "SetA").mkdir(parents=True)
    stale = sr_dir / "SetA" / "stale.png"
    stale.write_bytes(b"old")

    with pytest.raises(FileExistsError):
        generate_sr_results(checkpoint, data_dir / "testsets", sr_dir, ["SetA"], device="cpu")

    assert stale.read_bytes() == b"old"  # 이전 실행의 파일을 섞거나 지우지 않는다.


def test_generate_sr_results_leaves_nothing_when_an_image_fails(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    (data_dir / "testsets" / "SetA" / "LR" / "img9.png").write_bytes(b"not a png")
    checkpoint = tmp_path / "ckpt" / "pilot.pth"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    out_root = tmp_path / "out_root"
    out_root.mkdir()

    with pytest.raises(Exception):
        generate_sr_results(checkpoint, data_dir / "testsets", out_root / "sr", ["SetA"], device="cpu")

    assert list(out_root.iterdir()) == []  # 반쯤 쓴 SR 결과·임시 폴더가 남지 않는다.


def test_train_to_pilot_diagnosis_end_to_end(data_dir: Path, tmp_path: Path) -> None:
    """학습 → SR 결과 → 진단 4종이 이어져 `results/pilot/`에 들어갈 산출물이 만들어진다."""
    _write_testsets(data_dir, ("SetA",), ((32, 32), (40, 24)))
    checkpoint, sr_dir, output_dir = tmp_path / "pilot.pth", tmp_path / "sr", tmp_path / "pilot_out"
    train_pilot_baseline(data_dir, checkpoint, **TRAIN_KWARGS)
    generate_sr_results(checkpoint, data_dir / "testsets", sr_dir, ["SetA"], device="cpu")
    fair_comparison = {"fixed_rules": {"shave_border": 4, "datasets": ["SetA"]}, "pilot_determined": {"edge_top_q": None}}

    settings = run_pilot_diagnosis(
        sr_dir, data_dir / "testsets", output_dir, fair_comparison, commit="test", edge_top_q=20.0
    )

    assert settings["shave_border"] == 4
    for name in (
        "per_image_diagnosis.csv",
        "summary_diagnosis.csv",
        "worst_images.csv",
        "diagnosis_settings.json",
        "edge_flat_psnr.png",
        "ycbcr_error_share.png",
        "spectrum_SetA.png",
    ):
        assert (output_dir / name).is_file()
    per_image = pd.read_csv(output_dir / "per_image_diagnosis.csv")
    assert len(per_image) == 2  # 테스트 영상 2장
    assert np.isfinite(per_image.drop(columns=["dataset", "image_id"]).to_numpy(dtype=float)).all()


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _load_cli(commit=lambda **kwargs: COMMIT):
    """CLI 모듈을 읽고 커밋 조회를 바꿔 끼운다(작업 트리가 dirty여도, git이 없어도 테스트가 돈다).

    `commit`은 `resolve_code_commit`에 넘어온 키워드 인자를 받아 기록할 커밋 문자열을 돌려주거나 예외를 낸다.
    넘어온 인자는 `module.commit_calls`에 쌓인다. 실제 git 동작은 `tests/test_provenance.py`가 검증한다.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pilot_baseline.py"
    spec = importlib.util.spec_from_file_location("train_pilot_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.commit_calls = []

    def resolve_code_commit(repo_root, **kwargs):
        module.commit_calls.append(kwargs)
        return commit(**kwargs)

    module.resolve_code_commit = resolve_code_commit
    return module


def _cli_args(data_dir: Path, tmp_path: Path, config: Path) -> list[str]:
    return [
        "--data-dir", str(data_dir),
        "--checkpoint-path", str(tmp_path / "ckpt" / "pilot.pth"),
        "--sr-dir", str(tmp_path / "sr"),
        "--output-dir", str(tmp_path / "out"),
        "--seed", "0", "--epochs", "1", "--patches-per-epoch", "8", "--batch-size", "4",
        "--learning-rate", "0.001", "--optimizer", "adam", "--scheduler", "none", "--augmentation", "rot90x4_hflip",
        "--device", "cpu",
        "--fair-comparison-config", str(config),
    ]


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_FAIR_COMPARISON = REPO_ROOT / "configs" / "fair_comparison.yaml"


def test_cli_with_repo_fair_comparison_writes_sr_results_and_settings_without_paths(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """실제 공정 비교 조건 파일의 테스트셋 4종 이름으로 끝까지 돈다. 기록·표준 출력에 경로가 없다."""
    with REPO_FAIR_COMPARISON.open(encoding="utf-8") as file:
        dataset_names = tuple(yaml.safe_load(file)["fixed_rules"]["datasets"])
    _write_testsets(data_dir, dataset_names, ((16, 16),))
    cli = _load_cli()

    cli.main(_cli_args(data_dir, tmp_path, REPO_FAIR_COMPARISON))

    for name in dataset_names:
        assert (tmp_path / "sr" / name / "img0.png").is_file()
    settings_text = (tmp_path / "out" / "pilot_train_settings.json").read_text(encoding="utf-8")
    settings = json.loads(settings_text)
    assert settings["commit"] == COMMIT
    assert cli.commit_calls == [{"allow_dirty": False}]
    assert settings["patch_pool_id"] == "synthetic_pool"
    assert (settings["optimizer"], settings["scheduler"], settings["augmentation"]) == ("adam", "none", "rot90x4_hflip")
    assert settings["sr_image_counts"] == {name: 1 for name in dataset_names}
    # Public 저장소에 커밋되고 노트북 출력에도 남으므로 경로를 기록·출력하지 않는다.
    stdout = capsys.readouterr().out
    for path in (tmp_path, data_dir):
        assert str(path) not in settings_text
        assert str(path) not in stdout


def _write_config(tmp_path: Path, text: str) -> Path:
    config = tmp_path / "fair.yaml"
    config.write_text(text, encoding="utf-8")
    return config


def test_cli_allow_dirty_is_passed_through_and_dirty_commit_is_recorded(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    config = _write_config(tmp_path, "fixed_rules:\n  datasets: [SetA]\n")
    cli = _load_cli(commit=lambda allow_dirty: f"{COMMIT}-dirty" if allow_dirty else COMMIT)

    cli.main([*_cli_args(data_dir, tmp_path, config), "--allow-dirty"])

    assert cli.commit_calls == [{"allow_dirty": True}]
    settings = json.loads((tmp_path / "out" / "pilot_train_settings.json").read_text(encoding="utf-8"))
    assert settings["commit"] == f"{COMMIT}-dirty"


def test_cli_stops_before_training_when_code_has_uncommitted_changes(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    config = _write_config(tmp_path, "fixed_rules:\n  datasets: [SetA]\n")

    def refuse(allow_dirty: bool) -> str:
        raise RuntimeError("코드·설정에 커밋되지 않은 변경이 있어")

    cli = _load_cli(commit=refuse)

    with pytest.raises(SystemExit, match="학습을 시작하지 않았습니다"):
        cli.main(_cli_args(data_dir, tmp_path, config))

    assert cli.commit_calls == [{"allow_dirty": False}]
    # 긴 학습 전에 멈춰야 하므로 체크포인트·SR 결과·학습 설정 기록이 하나도 없다.
    assert not (tmp_path / "ckpt").exists()
    assert not (tmp_path / "sr").exists()
    assert not (tmp_path / "out").exists()


def test_cli_checks_testsets_before_training(data_dir: Path, tmp_path: Path) -> None:
    config = _write_config(tmp_path, "fixed_rules:\n  datasets: [MissingSet]\n")

    with pytest.raises(ValueError):
        _load_cli().main(_cli_args(data_dir, tmp_path, config))

    assert not (tmp_path / "ckpt" / "pilot.pth").exists()  # 긴 학습 전에 실패해야 한다.


def test_cli_refuses_existing_sr_dir_before_training(data_dir: Path, tmp_path: Path) -> None:
    _write_testsets(data_dir, ("SetA",), ((16, 16),))
    config = _write_config(tmp_path, "fixed_rules:\n  datasets: [SetA]\n")
    (tmp_path / "sr").mkdir()

    with pytest.raises(FileExistsError):
        _load_cli().main(_cli_args(data_dir, tmp_path, config))

    assert not (tmp_path / "ckpt" / "pilot.pth").exists()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "fixed_rules: []\n",
        "fixed_rules:\n  scale: 4\n",
        "fixed_rules:\n  datasets: []\n",
        "fixed_rules:\n  datasets: SetA\n",
        "fixed_rules:\n  datasets: [1, 2]\n",
    ],
)
def test_cli_rejects_malformed_fair_comparison_before_training(data_dir: Path, tmp_path: Path, text: str) -> None:
    config = _write_config(tmp_path, text)

    with pytest.raises(ValueError):
        _load_cli().main(_cli_args(data_dir, tmp_path, config))

    assert not (tmp_path / "ckpt" / "pilot.pth").exists()


@pytest.mark.parametrize("flag", ["--epochs", "--optimizer", "--scheduler", "--augmentation"])
def test_cli_requires_training_settings(data_dir: Path, tmp_path: Path, flag: str) -> None:
    args = _cli_args(data_dir, tmp_path, tmp_path / "fair.yaml")
    index = args.index(flag)
    del args[index : index + 2]

    with pytest.raises(SystemExit):
        _load_cli().main(args)
