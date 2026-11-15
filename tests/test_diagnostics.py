"""진단 5종 바깥 동작 테스트 (SPEC.md US52~56, Task T2.3).

작은 합성 영상만 쓰므로 로컬 CPU에서 돈다. 공개 함수의 반환값과 저장 파일만 검증한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sr_pipeline.common.color import rgb_to_y, rgb_to_ycbcr, ycbcr_to_rgb
from sr_pipeline.diagnosis.diagnostics import (
    compute_edge_flat_psnr,
    compute_error_map,
    compute_frequency_spectrum_diff,
    decompose_ycbcr_error,
    rank_worst_images,
    run_diagnosis_suite,
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
