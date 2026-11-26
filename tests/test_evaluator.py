"""evaluate_model 경계 테스트: 모델 + 실행 정보 + 테스트셋 → summary·per_image 기록 (SPEC Testing Decisions §1·§2)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from sr_pipeline.common.bicubic import bicubic_resize
from sr_pipeline.data.prepare import prepare_testsets
from sr_pipeline.evaluation.evaluator import PerImageRecord, SummaryRecord, evaluate_model
from sr_pipeline.models.espcn import ESPCN

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Set5", "Set14", "BSD100", "Urban100")
SIZES = ((32, 40), (28, 28))  # 가장자리 4화소를 자른 뒤에도 SSIM 창(11)보다 크다.
METRICS = ("psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips")


class StubLpips(torch.nn.Module):
    """실제 AlexNet 가중치 없이 쓰는 LPIPS 대역: 입력 차이의 평균 제곱."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).pow(2).mean().view(1, 1, 1, 1)


class BicubicAsModel(torch.nn.Module):
    """학습 없이 bicubic 확대를 그대로 내는 모델: `model=None` 경로와 같은 지표가 나와야 한다."""

    def _network(self, x: torch.Tensor) -> torch.Tensor:  # 추론 시간의 "네트워크만" 측정용(신경망 없음)
        return x

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return bicubic_resize(lr_rgb, 4.0)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("eval_data")
    rng = np.random.default_rng(0)
    hr_root = root / "hr_src"
    for name in DATASETS:
        (hr_root / name).mkdir(parents=True)
        for index, (h, w) in enumerate(SIZES):
            # 가로 그라디언트 + 잡음: 에지와 평탄 영역이 모두 있다.
            base = np.linspace(0, 255, w, dtype=np.float64)[None, :, None].repeat(h, 0).repeat(3, 2)
            noisy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(hr_root / name / f"img{index}.png")
    out = root / "data"
    prepare_testsets(hr_root, out, 4, list(DATASETS), "bicubic", "matlab_imresize_antialias")
    return out


def fair(*, representative: dict | None = None, **pilot: Any) -> dict[str, Any]:
    config = yaml.safe_load((REPO_ROOT / "configs" / "fair_comparison.yaml").read_text(encoding="utf-8"))
    config["pilot_determined"]["edge_top_q"] = 10
    config["pilot_determined"].update(pilot)
    if representative is not None:
        config["pilot_determined"]["representative_images"] = representative
    config["infer_benchmark"].update(lr_height=12, lr_width=16, sr_height=48, sr_width=64, warmup_runs=1, repeat_runs=2)
    return copy.deepcopy(config)


def run_meta(config_id: str = "baseline", switches: dict | None = None, **overrides: Any) -> dict[str, Any]:
    meta = {
        "run_id": f"c0ffee-{config_id}-s0",
        "commit": "c0ffee",
        "config_id": config_id,
        "seed": 0,
        "patch_pool_id": "pool",
        "switches": {"A": False, "B": False, "C": False, "Bp": False, **(switches or {})},
        "params": 123,
        "train_minutes": 1.5,
        "status": "complete",
    }
    meta.update(overrides)
    return meta


def evaluate(model, tmp_path, data_dir, config=None, meta=None):
    return evaluate_model(
        model,
        meta or run_meta(),
        data_dir,
        tmp_path / "run",
        fair_comparison=config or fair(),
        lpips_model=StubLpips(),
    )


def test_one_summary_row_per_dataset_and_one_per_image_row_per_image(tmp_path: Path, data_dir: Path) -> None:
    summaries, per_images = evaluate(ESPCN().eval(), tmp_path, data_dir)

    assert [s.dataset for s in summaries] == list(DATASETS)
    assert len(per_images) == len(DATASETS) * len(SIZES)
    assert all(isinstance(s, SummaryRecord) for s in summaries)
    assert all(isinstance(p, PerImageRecord) for p in per_images)
    assert {p.image_id for p in per_images} == {"img0", "img1"}
    first = summaries[0]
    assert (first.run_id, first.commit, first.config_id, first.seed) == ("c0ffee-baseline-s0", "c0ffee", "baseline", 0)
    assert (first.patch_pool_id, first.switches, first.params, first.status) == ("pool", "none", 123, "complete")
    assert first.train_minutes == 1.5
    assert first.infer_ms_net > 0 and first.infer_ms_total > 0


def test_summary_is_mean_of_per_image_values(tmp_path: Path, data_dir: Path) -> None:
    summaries, per_images = evaluate(ESPCN().eval(), tmp_path, data_dir)
    for summary in summaries:
        rows = [p for p in per_images if p.dataset == summary.dataset]
        for metric in METRICS:
            assert getattr(summary, metric) == pytest.approx(np.mean([getattr(r, metric) for r in rows]), abs=1e-12)


def test_summary_and_per_image_files_are_written_with_spec_columns(tmp_path: Path, data_dir: Path) -> None:
    evaluate(ESPCN(use_y_channel=True).eval(), tmp_path, data_dir, meta=run_meta("B", {"B": True}))
    summary = pd.read_csv(tmp_path / "run" / "summary.csv", dtype={"switches": str})
    per_image = pd.read_csv(tmp_path / "run" / "per_image.csv", dtype={"image_id": str})

    assert list(summary.columns) == [
        "run_id", "commit", "config_id", "seed", "patch_pool_id", "switches", "dataset", "psnr_y", "ssim_y",
        "psnr_rgb", "edge_psnr", "lpips", "psnr_y_pre_rgb", "params", "infer_ms_net", "infer_ms_total",
        "train_minutes", "status",
    ]  # fmt: skip
    assert list(per_image.columns) == [
        "run_id", "dataset", "image_id", "psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips",
    ]  # fmt: skip
    assert len(summary) == 4 and len(per_image) == 8
    assert set(summary["switches"]) == {"B"}


def test_every_config_goes_through_the_same_evaluation_path(tmp_path: Path, data_dir: Path) -> None:
    """bicubic을 그대로 내는 모델과 학습 없는 구성(model=None)은 시간 외 모든 지표가 같다."""
    via_model, per_model = evaluate_model(
        BicubicAsModel().eval(), run_meta("x"), data_dir, tmp_path / "a", fair_comparison=fair(), lpips_model=StubLpips()
    )
    via_none, per_none = evaluate_model(
        None, run_meta("x"), data_dir, tmp_path / "b", fair_comparison=fair(), lpips_model=StubLpips()
    )
    assert per_model == per_none
    for left, right in zip(via_model, via_none):
        for metric in METRICS:
            assert getattr(left, metric) == getattr(right, metric)
    assert all(s.infer_ms_net == 0.0 and s.infer_ms_total > 0 for s in via_none)


def test_psnr_y_pre_rgb_is_filled_only_for_b_family(tmp_path: Path, data_dir: Path) -> None:
    b_rows, _ = evaluate(ESPCN(use_y_channel=True).eval(), tmp_path / "b", data_dir, meta=run_meta("B", {"B": True}))
    rgb_rows, _ = evaluate(ESPCN().eval(), tmp_path / "rgb", data_dir)
    bp_rows, _ = evaluate(ESPCN().eval(), tmp_path / "bp", data_dir, meta=run_meta("Bp", {"Bp": True}))
    none_rows, _ = evaluate(None, tmp_path / "none", data_dir, meta=run_meta("bicubic"))

    assert all(s.psnr_y_pre_rgb is not None and math.isfinite(s.psnr_y_pre_rgb) for s in b_rows)
    assert all(s.psnr_y_pre_rgb is None for rows in (rgb_rows, bp_rows, none_rows) for s in rows)


def test_psnr_y_pre_rgb_is_close_to_psnr_y(tmp_path: Path, data_dir: Path) -> None:
    """RGB로 되돌려도 Y는 거의 그대로이므로 진단값은 PSNR(Y)과 가까워야 한다(자르기·양자화 차이만)."""
    rows, _ = evaluate(ESPCN(use_y_channel=True).eval(), tmp_path, data_dir, meta=run_meta("B", {"B": True}))
    for s in rows:
        assert s.psnr_y_pre_rgb == pytest.approx(s.psnr_y, abs=1.0)


def test_switches_are_written_as_plus_joined_enabled_names(tmp_path: Path, data_dir: Path) -> None:
    rows, _ = evaluate(
        ESPCN(use_y_channel=True, use_global_residual=True).eval(),
        tmp_path,
        data_dir,
        meta=run_meta("ABC", {"A": True, "B": True, "C": True}),
    )
    assert {s.switches for s in rows} == {"A+B+C"}


def test_shave_border_comes_from_fair_comparison(tmp_path: Path, data_dir: Path) -> None:
    border4, _ = evaluate(BicubicAsModel().eval(), tmp_path / "a", data_dir)
    config0 = fair()
    config0["fixed_rules"]["shave_border"] = 0
    border0, _ = evaluate(BicubicAsModel().eval(), tmp_path / "b", data_dir, config=config0)
    assert border4[0].psnr_y != border0[0].psnr_y


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["pilot_determined"].update(edge_top_q=None), "edge_top_q"),
        (lambda c: c["fixed_rules"].update(lpips_backbone="vgg"), "lpips"),
        (lambda c: c["infer_benchmark"].update(precision="fp16"), "precision"),
        (lambda c: c["infer_benchmark"].update(sr_width=1281), "sr_width"),
    ],
)
def test_invalid_fair_conditions_are_rejected_before_writing(tmp_path: Path, data_dir: Path, mutate, message: str) -> None:
    config = fair()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        evaluate(ESPCN().eval(), tmp_path, data_dir, config=config)
    assert not (tmp_path / "run" / "summary.csv").exists()


def test_existing_records_are_not_overwritten(tmp_path: Path, data_dir: Path) -> None:
    evaluate(ESPCN().eval(), tmp_path, data_dir)
    with pytest.raises(FileExistsError):
        evaluate(ESPCN().eval(), tmp_path, data_dir)


def test_run_meta_missing_fields_are_rejected(tmp_path: Path, data_dir: Path) -> None:
    meta = run_meta()
    del meta["params"]
    with pytest.raises(ValueError, match="params"):
        evaluate(ESPCN().eval(), tmp_path, data_dir, meta=meta)


def test_representative_images_are_saved_when_specified_and_skipped_when_null(tmp_path: Path, data_dir: Path) -> None:
    null_rep = {name: {"median": None, "worst": None} for name in DATASETS}
    evaluate(ESPCN().eval(), tmp_path / "skip", data_dir, config=fair(representative=null_rep))
    assert not (tmp_path / "skip" / "run" / "images").exists()

    rep = {name: {"median": "img0", "worst": "img1"} for name in DATASETS}
    evaluate(ESPCN().eval(), tmp_path / "save", data_dir, config=fair(representative=rep))
    evaluate(BicubicAsModel().eval(), tmp_path / "other", data_dir, config=fair(representative=rep))
    images = tmp_path / "save" / "run" / "images" / "Set5"
    full = np.array(Image.open(images / "img0_median.png"))
    zoom = np.array(Image.open(images / "img0_median_roi.png"))
    assert full.shape == (32, 3 * 40, 3)  # Before | After | HR, 원래 HR 크기
    assert zoom.ndim == 3 and zoom.shape[1] % 3 == 0
    assert (images / "img1_worst.png").is_file() and (images / "img1_worst_roi.png").is_file()
    # Before(bicubic 확대)와 HR 패널은 모델과 무관하게 같고, After 패널은 모델마다 다르다.
    other = np.array(Image.open(tmp_path / "other" / "run" / "images" / "Set5" / "img0_median.png"))
    assert np.array_equal(full[:, :40], other[:, :40]) and np.array_equal(full[:, 80:], other[:, 80:])
    assert np.array_equal(other[:, :40], other[:, 40:80])  # bicubic 모델의 After는 Before와 같다
    assert not np.array_equal(full[:, 40:80], other[:, 40:80])


def test_unknown_representative_image_is_rejected_before_writing(tmp_path: Path, data_dir: Path) -> None:
    rep = {name: {"median": "nope", "worst": None} for name in DATASETS}
    with pytest.raises(ValueError, match="nope"):
        evaluate(ESPCN().eval(), tmp_path, data_dir, config=fair(representative=rep))
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").iterdir())


def test_evaluation_leaves_model_weights_untouched(tmp_path: Path, data_dir: Path) -> None:
    model = ESPCN().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate(model, tmp_path, data_dir)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
