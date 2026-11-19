"""진단 5종 바깥 동작 테스트 (SPEC.md US52~56, Task T2.3).

작은 합성 영상만 쓰므로 로컬 CPU에서 돈다. 공개 함수의 반환값과 저장 파일만 검증한다.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from sr_pipeline.common.color import rgb_to_y, rgb_to_ycbcr, ycbcr_to_rgb
from sr_pipeline.diagnosis.diagnostics import (
    compute_edge_flat_psnr,
    compute_error_map,
    compute_frequency_spectrum_diff,
    decompose_ycbcr_error,
    rank_worst_images,
    resolve_diagnosis_settings,
    run_diagnosis_suite,
    run_pilot_diagnosis,
)


def _rgb(height: int = 32, width: int = 32, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, height, width, generator=generator)


def _blur(image: torch.Tensor, size: int = 5) -> torch.Tensor:
    return F.avg_pool2d(image, size, stride=1, padding=size // 2, count_include_pad=False)


def _vertical_step_y(height: int = 32, width: int = 32) -> torch.Tensor:
    """왼쪽 Y=50, 오른쪽 Y=200인 세로 계단 에지 영상 (1, 1, H, W)."""
    y = torch.full((1, 1, height, width), 50.0)
    y[..., width // 2 :] = 200.0
    return y


# ---------------------------------------------------------------- 오차 지도


def test_error_map_is_zero_for_identical_images() -> None:
    hr = _rgb(12, 20)
    error = compute_error_map(hr.clone(), hr)
    assert error.shape == (12, 20)
    assert (error == 0).all()


def test_error_map_marks_only_changed_pixel() -> None:
    hr = _rgb()
    sr = hr.clone()
    sr[0, :, 5, 7] += 0.2
    error = compute_error_map(sr, hr)
    assert error[5, 7] > 0
    error[5, 7] = 0
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-4)


# ---------------------------------------------------------------- 에지·평탄 PSNR(Y)


def test_edge_flat_psnr_is_inf_for_identical_images() -> None:
    hr_y = _vertical_step_y()
    result = compute_edge_flat_psnr(hr_y.clone(), hr_y, top_q_percent=20.0)
    assert result == {"edge_psnr_y": math.inf, "flat_psnr_y": math.inf}


def test_blurred_edge_has_lower_edge_psnr_than_flat_psnr() -> None:
    hr_y = _vertical_step_y()
    result = compute_edge_flat_psnr(_blur(hr_y), hr_y, top_q_percent=5.0)
    assert math.isfinite(result["edge_psnr_y"])
    assert result["edge_psnr_y"] < result["flat_psnr_y"]


@pytest.mark.parametrize("q", [0.0, 100.0, -1.0])
def test_edge_flat_psnr_rejects_invalid_q(q: float) -> None:
    hr_y = _vertical_step_y()
    with pytest.raises(ValueError):
        compute_edge_flat_psnr(hr_y, hr_y, top_q_percent=q)


# ---------------------------------------------------------------- 주파수 스펙트럼


def test_spectrum_band_ratios_are_one_for_identical_images() -> None:
    hr_y = rgb_to_y(_rgb())
    result = compute_frequency_spectrum_diff(hr_y.clone(), hr_y)
    assert result["band_energy_ratio"] == pytest.approx({"low": 1.0, "mid": 1.0, "high": 1.0})
    assert result["sr_log_spectrum"].shape == (32, 32)
    assert len(result["radial_freq"]) == len(result["sr_radial"]) == len(result["hr_radial"])


def _periodic_y(size: int) -> torch.Tensor:
    """평균 밝기 120, 주기 8화소 세로 줄무늬 (1, 1, size, size). 주파수 1/8은 32·64 모두 정확한 FFT 구간이다."""
    x = torch.arange(size, dtype=torch.float32)
    stripes = 120.0 + 40.0 * torch.cos(2 * math.pi * x / 8)
    return stripes.expand(size, size).reshape(1, 1, size, size).clone()


def test_spectrum_does_not_depend_on_image_size() -> None:
    """같은 밝기·무늬면 영상 크기가 달라도 스펙트럼이 같아야 테스트셋 평균에서 큰 영상이 과대 반영되지 않는다."""
    small, large = (compute_frequency_spectrum_diff(y, y)["hr_radial"] for y in (_periodic_y(32), _periodic_y(64)))
    assert np.allclose(small, large, rtol=1e-4, atol=1e-3)
    assert small[0] == pytest.approx(120.0**2, rel=1e-6)  # DC 구간 = 평균 밝기²
    assert np.sort(small)[-2] == pytest.approx(2 * 20.0**2, rel=1e-4)  # 진폭 40 cos → ±1/8에 20씩


def test_blur_loses_high_frequency_energy() -> None:
    hr_y = rgb_to_y(_rgb(64, 64))
    ratio = compute_frequency_spectrum_diff(_blur(hr_y, 3), hr_y)["band_energy_ratio"]
    assert ratio["high"] < 0.5
    assert ratio["low"] > 0.8
    assert ratio["high"] < ratio["mid"] < ratio["low"]


# ---------------------------------------------------------------- YCbCr 분해


def test_luminance_only_error_goes_to_y() -> None:
    hr = _rgb() * 0.8 + 0.1
    result = decompose_ycbcr_error(hr + 0.05, hr)  # R=G=B 같은 양 → Cb·Cr 그대로
    assert result["share_y"] == pytest.approx(1.0, abs=1e-6)
    assert result["share_cb"] + result["share_y"] + result["share_cr"] == pytest.approx(1.0)


def test_chroma_only_error_goes_to_cb() -> None:
    hr = _rgb() * 0.5 + 0.25
    ycbcr = rgb_to_ycbcr(hr)
    ycbcr[:, 1] += 10.0
    result = decompose_ycbcr_error(ycbcr_to_rgb(ycbcr), hr)
    assert result["share_cb"] == pytest.approx(1.0, abs=1e-4)
    assert result["share_y"] == pytest.approx(0.0, abs=1e-4)


def test_ycbcr_shares_are_zero_without_error() -> None:
    hr = _rgb()
    result = decompose_ycbcr_error(hr.clone(), hr)
    assert result["mse_y"] == result["share_y"] == 0.0


# ---------------------------------------------------------------- 최악 영상 목록


def test_rank_worst_images_per_dataset(tmp_path: Path) -> None:
    csv = tmp_path / "per_image.csv"
    pd.DataFrame(
        {
            "run_id": ["r"] * 6,
            "dataset": ["Set5", "Set5", "Set5", "Set14", "Set14", "Set14"],
            "image_id": ["c", "a", "b", "x", "y", "z"],
            "psnr_y": [30.0, 25.0, 25.0, 28.0, 20.0, 35.0],
        }
    ).to_csv(csv, index=False)
    worst = rank_worst_images(csv, top_k=2)
    assert [(w["dataset"], w["image_id"], w["rank"]) for w in worst] == [
        ("Set14", "y", 1),
        ("Set14", "x", 2),
        ("Set5", "a", 1),  # 동점(25.0)은 image_id 순서
        ("Set5", "b", 2),
    ]


def test_rank_worst_images_rejects_mixed_runs(tmp_path: Path) -> None:
    csv = tmp_path / "per_image.csv"
    pd.DataFrame({"run_id": ["r1", "r2"], "dataset": ["Set5"] * 2, "image_id": ["a", "b"], "psnr_y": [1.0, 2.0]}).to_csv(
        csv, index=False
    )
    with pytest.raises(ValueError):
        rank_worst_images(csv)


# ---------------------------------------------------------------- 잘못된 입력


@pytest.mark.parametrize(
    "sr, hr",
    [
        (torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)),
        (torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 9)),
        (torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8), torch.rand(1, 3, 8, 8)),
    ],
    ids=["batch-2", "shape-mismatch", "uint8"],
)
def test_rejects_invalid_rgb_input(sr: torch.Tensor, hr: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        compute_error_map(sr, hr)
    with pytest.raises(ValueError):
        decompose_ycbcr_error(sr, hr)


# ---------------------------------------------------------------- 전체 실행 (완료 조건: 5종 산출물)


def _write_png(path: Path, rgb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (rgb[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def test_run_diagnosis_suite_writes_all_outputs(tmp_path: Path) -> None:
    hr_root, sr_root, out = tmp_path / "testsets", tmp_path / "sr", tmp_path / "out"
    for dataset in ("Set5", "Set14"):
        for index in range(3):
            hr = _rgb(40, 48, seed=index)
            _write_png(hr_root / dataset / "HR" / f"img{index}.png", hr)
            _write_png(sr_root / dataset / f"img{index}.png", _blur(hr, 3))

    run_diagnosis_suite(sr_root, hr_root, out, top_q_percent=20.0, shave_border=4)

    per_image = pd.read_csv(out / "per_image_diagnosis.csv")
    assert len(per_image) == 6
    assert {"dataset", "image_id", "psnr_y", "edge_psnr_y", "flat_psnr_y", "share_y", "band_ratio_high"} <= set(
        per_image.columns
    )
    assert (per_image["band_ratio_high"] < 1).all()
    assert sorted(pd.read_csv(out / "summary_diagnosis.csv")["dataset"]) == ["Set14", "Set5"]
    assert len(pd.read_csv(out / "worst_images.csv")) == 6  # 테스트셋당 3장 < top_k 5
    for name in ("spectrum_Set5.png", "spectrum_Set14.png", "edge_flat_psnr.png", "ycbcr_error_share.png"):
        assert (out / name).is_file(), name
    assert len(list((out / "error_maps").glob("*.png"))) == 6


def test_run_diagnosis_suite_requires_matching_sr(tmp_path: Path) -> None:
    hr_root = tmp_path / "testsets"
    _write_png(hr_root / "Set5" / "HR" / "img0.png", _rgb())
    (tmp_path / "sr" / "Set5").mkdir(parents=True)
    with pytest.raises(ValueError):
        run_diagnosis_suite(tmp_path / "sr", hr_root, tmp_path / "out", top_q_percent=20.0, shave_border=4)


def test_run_diagnosis_suite_requires_explicit_fair_comparison_values(tmp_path: Path) -> None:
    """edge_top_q·shave_border는 fair_comparison.yaml 값이라 기본값 없이 호출하는 쪽이 넘겨야 한다."""
    with pytest.raises(TypeError):
        run_diagnosis_suite(tmp_path / "sr", tmp_path / "testsets", tmp_path / "out")  # type: ignore[call-arg]


# ---------------------------------------------------------------- 파일럿 진단 CLI (T2.5)

REPO_ROOT = Path(__file__).resolve().parents[1]


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _fair_comparison(
    edge_top_q: float | None, shave_border: int | None = 4, datasets: object = ("Set5",)
) -> dict:
    fixed_rules: dict = {} if shave_border is None else {"shave_border": shave_border}
    fixed_rules["datasets"] = list(datasets) if isinstance(datasets, tuple) else datasets
    return {"fixed_rules": fixed_rules, "pilot_determined": {"edge_top_q": edge_top_q}}


def test_settings_use_fair_comparison_values_when_fixed() -> None:
    settings = resolve_diagnosis_settings(_fair_comparison(15.0, shave_border=6), edge_top_q=None)
    assert settings == {
        "edge_top_q": 15.0,
        "edge_top_q_source": "fair_comparison",
        "shave_border": 6,
        "datasets": ["Set5"],
    }


def test_settings_accept_matching_command_line_q_when_fixed() -> None:
    settings = resolve_diagnosis_settings(_fair_comparison(15.0), edge_top_q=15.0)
    assert settings["edge_top_q_source"] == "fair_comparison"


def test_settings_require_command_line_q_while_unfixed() -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(None), edge_top_q=None)


def test_settings_use_command_line_q_while_unfixed() -> None:
    settings = resolve_diagnosis_settings(_fair_comparison(None), edge_top_q=20.0)
    assert settings == {
        "edge_top_q": 20.0,
        "edge_top_q_source": "command_line",
        "shave_border": 4,
        "datasets": ["Set5"],
    }


def test_settings_reject_command_line_q_different_from_fixed() -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(15.0), edge_top_q=20.0)


@pytest.mark.parametrize("q", [0.0, 100.0, -1.0])
def test_settings_reject_invalid_q(q: float) -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(None), edge_top_q=q)


def test_settings_require_shave_border_in_fixed_rules() -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(None, shave_border=None), edge_top_q=20.0)


def _load_pilot_script():
    path = REPO_ROOT / "scripts" / "run_pilot_diagnosis.py"
    spec = importlib.util.spec_from_file_location("run_pilot_diagnosis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pilot_pairs(
    tmp_path: Path, datasets: tuple[str, ...] = ("Set5",), count: int = 2
) -> tuple[Path, Path, Path]:
    hr_root, sr_root, out = tmp_path / "testsets", tmp_path / "sr", tmp_path / "out"
    for dataset in datasets:
        for index in range(count):
            hr = _rgb(40, 48, seed=index)
            _write_png(hr_root / dataset / "HR" / f"img{index}.png", hr)
            _write_png(sr_root / dataset / f"img{index}.png", _blur(hr, 3))
    return sr_root, hr_root, out


def _run_pilot_cli(sr_root: Path, hr_root: Path, out: Path, config: Path, edge_top_q: str | None) -> dict:
    argv = ["--sr-dir", str(sr_root), "--hr-dir", str(hr_root), "--output-dir", str(out)]
    argv += ["--fair-comparison-config", str(config)]
    if edge_top_q is not None:
        argv += ["--edge-top-q", edge_top_q]
    script = _load_pilot_script()
    # git이 없거나 .git 없이 받은 저장소에서도 테스트가 돌도록 커밋 조회는 바꿔 끼운다.
    calls: list[dict] = []
    script.resolve_code_commit = lambda repo_root, **kwargs: calls.append(kwargs) or COMMIT
    script.main(argv)
    assert calls == [{"allow_dirty": False}]
    return json.loads((out / "diagnosis_settings.json").read_text(encoding="utf-8"))


@pytest.fixture
def system_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """진단 작업 폴더가 만들어지는 시스템 임시 폴더를 테스트 전용 폴더로 바꾼다."""
    path = tmp_path / "system_tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def test_pilot_diagnosis_cli_records_command_line_q_while_unfixed(tmp_path: Path) -> None:
    """edge_top_q가 null인 설정 파일(테스트 안에서 만든 임시 YAML)로 CLI를 끝까지 실행한다."""
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path)
    config = tmp_path / "fair_comparison.yaml"
    config.write_text(yaml.safe_dump(_fair_comparison(None)), encoding="utf-8")

    settings = _run_pilot_cli(sr_root, hr_root, out, config, edge_top_q="20")

    assert len(pd.read_csv(out / "per_image_diagnosis.csv")) == 2
    # 공개 저장소에 커밋되는 파일이라 입력 경로는 남기지 않고, 코드 커밋으로 실행을 추적한다.
    assert settings == {
        "edge_top_q": 20.0,
        "edge_top_q_source": "command_line",
        "shave_border": 4,
        "datasets": ["Set5"],
        "commit": COMMIT,
    }


def test_pilot_diagnosis_cli_runs_with_repo_fair_comparison_config(tmp_path: Path) -> None:
    """실제 configs/fair_comparison.yaml로 실행한다. edge_top_q 확정(T4.2) 전후 모두 통과해야 한다."""
    config = REPO_ROOT / "configs" / "fair_comparison.yaml"
    fair_comparison = yaml.safe_load(config.read_text(encoding="utf-8"))
    unfixed = fair_comparison["pilot_determined"].get("edge_top_q") is None
    datasets = tuple(fair_comparison["fixed_rules"]["datasets"])
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path, datasets=datasets, count=1)

    settings = _run_pilot_cli(sr_root, hr_root, out, config, edge_top_q="20" if unfixed else None)

    assert settings["shave_border"] == fair_comparison["fixed_rules"]["shave_border"]
    assert settings["datasets"] == list(datasets)


def _run_pilot(sr_root: Path, hr_root: Path, out: Path, **overrides: object) -> dict:
    kwargs = {"commit": COMMIT, "edge_top_q": 20.0, **overrides}
    return run_pilot_diagnosis(sr_root, hr_root, out, _fair_comparison(None), **kwargs)  # type: ignore[arg-type]


def _snapshot(directory: Path) -> dict[Path, bytes]:
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_pilot_rerun_replaces_only_diagnosis_outputs(tmp_path: Path, system_tmp: Path) -> None:
    """같은 폴더에 다시 돌리면 이전 진단 결과(오차 지도 포함)는 사라지고, 진단이 만들지 않은 파일은 남는다."""
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path, count=3)
    _run_pilot(sr_root, hr_root, out)
    (out / ".gitkeep").write_text("", encoding="utf-8")
    (out / "pilot_train_settings.json").write_text('{"seed": 0}', encoding="utf-8")
    (out / "spectrum_pilot_train_loss.png").write_bytes(b"not a diagnosis output")
    (out / "error_maps" / "notes.png").write_bytes(b"hand-picked")
    assert len(list((out / "error_maps").glob("Set5_*.png"))) == 3

    for index in (1, 2):  # 두 번째 실행은 영상 1장만
        (hr_root / "Set5" / "HR" / f"img{index}.png").unlink()
    _run_pilot(sr_root, hr_root, out, edge_top_q=10.0)

    assert sorted(path.name for path in (out / "error_maps").iterdir()) == ["Set5_img0.png", "notes.png"]
    assert len(pd.read_csv(out / "per_image_diagnosis.csv")) == 1
    assert json.loads((out / "diagnosis_settings.json").read_text(encoding="utf-8"))["edge_top_q"] == 10.0
    assert (out / ".gitkeep").is_file()
    assert (out / "pilot_train_settings.json").read_text(encoding="utf-8") == '{"seed": 0}'
    assert (out / "spectrum_pilot_train_loss.png").read_bytes() == b"not a diagnosis output"
    # 작업 폴더는 출력 폴더 옆(results/ 안)이 아니라 시스템 임시 폴더에 만들고, 끝나면 지운다.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "sr", "system_tmp", "testsets"]
    assert list(system_tmp.iterdir()) == []


def test_pilot_failure_leaves_previous_outputs_untouched(
    tmp_path: Path, system_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """진단 도중 실패하면 이전 결과(CSV·그림·설정 기록)가 한 바이트도 바뀌지 않는다."""
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path)
    _run_pilot(sr_root, hr_root, out)
    before = _snapshot(out)

    def broken_figures(*args: object, **kwargs: object) -> None:
        raise RuntimeError("그림 저장 실패")

    monkeypatch.setattr("sr_pipeline.diagnosis.diagnostics._save_figures", broken_figures)
    with pytest.raises(RuntimeError):
        _run_pilot(sr_root, hr_root, out, edge_top_q=10.0)

    assert _snapshot(out) == before
    assert list(system_tmp.iterdir()) == []


@pytest.mark.parametrize(
    "fails",
    [
        lambda src, dst: "staging" in Path(src).parts and Path(src).name == "summary_diagnosis.csv",
        lambda src, dst: "backup" in Path(dst).parts and Path(src).name.startswith("Set5_"),
    ],
    ids=["moving_new_outputs_in", "moving_old_outputs_aside"],
)
def test_pilot_swap_failure_restores_previous_outputs(
    tmp_path: Path, system_tmp: Path, monkeypatch: pytest.MonkeyPatch, fails
) -> None:
    """이전 결과를 치우거나 새 결과를 넣는 도중에 실패해도 이전 결과가 그대로 돌아온다."""
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path)
    _run_pilot(sr_root, hr_root, out)
    before = _snapshot(out)
    real_move = shutil.move

    def flaky_move(src: str, dst: str, *args: object, **kwargs: object):
        if fails(src, dst):
            raise PermissionError("파일이 다른 프로그램에서 열려 있음")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", flaky_move)
    with pytest.raises(PermissionError):
        _run_pilot(sr_root, hr_root, out, edge_top_q=10.0)

    assert _snapshot(out) == before
    assert list(system_tmp.iterdir()) == []


def test_pilot_rejects_empty_hr_folder(tmp_path: Path) -> None:
    """설정에 있는 테스트셋의 HR 폴더가 비어 있으면(압축 해제 실패 등) 결과에서 조용히 빠지지 않고 멈춘다."""
    sr_root, hr_root, out = tmp_path / "sr", tmp_path / "testsets", tmp_path / "out"
    (hr_root / "Set5" / "HR").mkdir(parents=True)
    with pytest.raises(ValueError):
        _run_pilot(sr_root, hr_root, out)
    assert not out.exists()


def test_run_diagnosis_suite_processes_only_given_datasets(tmp_path: Path) -> None:
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path, datasets=("Set5", "Set14"))
    run_diagnosis_suite(sr_root, hr_root, out, top_q_percent=20.0, shave_border=4, datasets=["Set5"])
    assert set(pd.read_csv(out / "per_image_diagnosis.csv")["dataset"]) == {"Set5"}


@pytest.mark.parametrize("hr_datasets", [(), ("Set5", "DIV2K_valid")])
def test_pilot_rejects_hr_datasets_different_from_fair_comparison(tmp_path: Path, hr_datasets: tuple[str, ...]) -> None:
    """HR 폴더의 데이터셋이 fixed_rules.datasets와 다르면(빠짐·추가) 아무것도 쓰기 전에 멈춘다."""
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path, datasets=hr_datasets or ("Set14",))
    with pytest.raises(ValueError):
        _run_pilot(sr_root, hr_root, out)
    assert not out.exists()


@pytest.mark.parametrize("datasets", [[], ["Set5", "Set5"], ["Set5", 14], "Set5", None])
def test_settings_reject_invalid_datasets(datasets: object) -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(None, datasets=datasets), edge_top_q=20.0)


@pytest.mark.parametrize("commit", ["", "   ", None])
def test_pilot_requires_commit(tmp_path: Path, commit: object) -> None:
    sr_root, hr_root, out = _write_pilot_pairs(tmp_path)
    with pytest.raises(ValueError):
        _run_pilot(sr_root, hr_root, out, commit=commit)
    assert not out.exists()


@pytest.mark.parametrize("shave_border", [None, True, 4.0, -1])
def test_settings_reject_invalid_shave_border(shave_border: object) -> None:
    fair_comparison = {
        "fixed_rules": {"shave_border": shave_border, "datasets": ["Set5"]},
        "pilot_determined": {"edge_top_q": None},
    }
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(fair_comparison, edge_top_q=20.0)


def test_settings_reject_bool_command_line_q() -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(None), edge_top_q=True)  # type: ignore[arg-type]


def test_settings_reject_bool_configured_q() -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(_fair_comparison(True), edge_top_q=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fair_comparison",
    [
        None,
        {"fixed_rules": None, "pilot_determined": {"edge_top_q": None}},
        {"fixed_rules": {"shave_border": 4}, "pilot_determined": []},
    ],
)
def test_settings_reject_malformed_config(fair_comparison: object) -> None:
    with pytest.raises(ValueError):
        resolve_diagnosis_settings(fair_comparison, edge_top_q=20.0)  # type: ignore[arg-type]


def test_pilot_diagnosis_writes_nothing_when_settings_invalid(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        run_pilot_diagnosis(
            tmp_path / "sr", tmp_path / "testsets", out, _fair_comparison(None), commit=COMMIT, edge_top_q=None
        )
    assert not out.exists()
