"""단일 평가 경로 통합 실행기 (SPEC.md / ADR-0003).

Bicubic 하한과 모든 학습 구성이 동일한 평가 경로를 통과하도록 강제합니다.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sr_pipeline.common.bicubic import bicubic_resize
from sr_pipeline.common.color import denormalize_y, rgb_to_y
from sr_pipeline.data.dataset import TestsetDataset
from sr_pipeline.evaluation.benchmark import measure_inference_time
from sr_pipeline.evaluation.metrics import (
    _check_top_q,
    _sobel_magnitude,
    compute_edge_psnr,
    compute_lpips,
    compute_psnr,
    compute_ssim_y,
)
from sr_pipeline.evaluation.postprocess import quantize_and_shave

SUMMARY_FILENAME = "summary.csv"
PER_IMAGE_FILENAME = "per_image.csv"
IMAGES_DIRNAME = "images"
_STAGING_DIRNAME = ".evaluation_tmp"
_RUN_META_KEYS = (
    "run_id",
    "commit",
    "config_id",
    "seed",
    "patch_pool_id",
    "switches",
    "params",
    "train_minutes",
    "status",
)
_SWITCH_ORDER = ("A", "B", "C", "Bp")
_STATUSES = ("complete", "incomplete")
_METRIC_KEYS = ("psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips")
ROI_SIZE_HR = 32  # 고주파 관심 영역 창(HR 화소). 영상이 작으면 영상 크기로 줄인다.
ROI_ZOOM = 4  # 확대본 배율(최근접, 화소 반복)


@dataclass(frozen=True)
class SummaryRecord:
    """실행 기록 summary 행 (실행 1회 x 테스트셋 1개, SPEC.md 준수)."""

    run_id: str
    commit: str
    config_id: str
    seed: int
    patch_pool_id: str
    switches: str
    dataset: str
    psnr_y: float
    ssim_y: float
    psnr_rgb: float
    edge_psnr: float
    lpips: float
    psnr_y_pre_rgb: float | None
    params: int
    infer_ms_net: float
    infer_ms_total: float
    train_minutes: float
    status: str  # "complete" | "incomplete"


@dataclass(frozen=True)
class PerImageRecord:
    """실행 기록 per_image 행 (실행 1회 x 테스트 영상 1장, SPEC.md 준수)."""

    run_id: str
    dataset: str
    image_id: str
    psnr_y: float
    ssim_y: float
    psnr_rgb: float
    edge_psnr: float
    lpips: float


def _switches_label(switches: dict[str, Any]) -> str:
    """켜진 스위치를 `A+B`처럼 잇고, 없으면 `none`(베이스라인·Bicubic 하한)."""
    unknown = set(switches) - set(_SWITCH_ORDER)
    if unknown:
        raise ValueError(f"run_meta['switches']에 알 수 없는 스위치가 있습니다: {sorted(unknown)}")
    enabled = [name for name in _SWITCH_ORDER if switches.get(name, False)]
    return "+".join(enabled) if enabled else "none"


def _check_inputs(run_meta: dict[str, Any], fair_comparison: dict[str, Any]) -> tuple[int, int, list[str], float, dict]:
    """평가에 필요한 값을 아무것도 쓰기 전에 검증하고 `(border, scale, datasets, edge_q, bench)`를 돌려준다."""
    missing = [key for key in _RUN_META_KEYS if key not in run_meta]
    if missing:
        raise ValueError(f"run_meta에 필요한 항목이 없습니다: {missing}")
    if run_meta["status"] not in _STATUSES:
        raise ValueError(f"run_meta['status']는 {_STATUSES} 중 하나여야 합니다: {run_meta['status']!r}")
    rules = fair_comparison["fixed_rules"]
    if rules["lpips_backbone"] != "alex":
        raise ValueError(f"lpips 백본은 alex로 고정입니다: {rules['lpips_backbone']!r}")
    edge_q = fair_comparison["pilot_determined"]["edge_top_q"]
    if edge_q is None:
        raise ValueError("pilot_determined.edge_top_q가 null입니다. T4.2에서 고정하기 전에는 평가할 수 없습니다.")
    _check_top_q(edge_q, "pilot_determined.edge_top_q")
    bench = fair_comparison["infer_benchmark"]
    if bench["precision"] != "fp32" or bench["batch_size"] != 1:
        raise ValueError(
            f"추론 시간 측정은 batch_size=1, precision=fp32로 고정입니다: "
            f"batch_size={bench['batch_size']!r}, precision={bench['precision']!r}"
        )
    scale = rules["scale"]
    for axis in ("height", "width"):
        if bench[f"sr_{axis}"] != bench[f"lr_{axis}"] * scale:
            raise ValueError(
                f"infer_benchmark의 sr_{axis}({bench[f'sr_{axis}']!r})가 lr_{axis}({bench[f'lr_{axis}']!r}) x 배율({scale})과 다릅니다."
            )
    return rules["shave_border"], rules["scale"], list(rules["datasets"]), edge_q, bench


def _super_resolve(model: nn.Module | None, lr: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """모든 구성이 지나는 SR 단계. 반환은 (자르기 전 RGB SR, B 계열의 RGB 복원 전 정규화 Y 또는 None)."""
    with torch.inference_mode():
        if model is None:  # Bicubic 하한: 학습 없는 구성
            return bicubic_resize(lr, float(scale)), None
        forward_raw = getattr(model, "forward_raw", None)
        if forward_raw is None:
            return model(lr), None
        out = forward_raw(lr)
        return out["sr_rgb"], out.get("sr_y_norm")


def _shave(y: torch.Tensor, border: int) -> torch.Tensor:
    return y[..., border:-border, border:-border] if border > 0 else y


def _pre_rgb_psnr(sr_y_norm: torch.Tensor, hr_y: torch.Tensor, border: int) -> float:
    """RGB로 되돌리기 전 Y의 PSNR(Y): 정규화 Y를 [0, 1]로 자르고 Y 스케일(16~235)로 복원 → 가장자리 자르기.

    `hr_y`(`common.color.rgb_to_y`)도 반올림하지 않은 Y이므로 여기서도 정수로 반올림하지 않는다.
    반올림하면 Y가 정확해도 약 59 dB에서 막혀, RGB 왕복의 자르기·양자화 영향(ADR-0003)을 잴 수 없다.
    """
    y = denormalize_y(sr_y_norm.clamp(0.0, 1.0))
    return compute_psnr(_shave(y, border), hr_y)


def _evaluate_image(
    model: nn.Module | None,
    sample: dict[str, Any],
    *,
    scale: int,
    border: int,
    edge_q: float,
    switch_b: bool,
    lpips_model: torch.nn.Module | None,
    device: torch.device,
) -> tuple[dict[str, float | None], torch.Tensor]:
    lr, hr = sample["lr"].to(device), sample["hr"].to(device)
    sr, sr_y_norm = _super_resolve(model, lr, scale)
    if model is not None and switch_b != (sr_y_norm is not None):
        raise ValueError(
            f"스위치 B={switch_b}인데 모델이 RGB로 되돌리기 전 Y(sr_y_norm)를 "
            f"{'내지 않습니다' if switch_b else '냅니다'}. run_meta['switches']와 모델 구성이 다릅니다."
        )
    if sr.shape != hr.shape:
        raise ValueError(f"{sample['image_id']}: SR 결과 {tuple(sr.shape)}가 HR {tuple(hr.shape)}와 다릅니다.")
    sr_q, hr_q = quantize_and_shave(sr, border=border), quantize_and_shave(hr, border=border)
    sr_y, hr_y = rgb_to_y(sr_q), rgb_to_y(hr_q)  # postprocess_to_y와 같은 정의
    metrics: dict[str, float | None] = {
        "psnr_y": compute_psnr(sr_y, hr_y),
        "ssim_y": compute_ssim_y(sr_y, hr_y),
        "psnr_rgb": compute_psnr(sr_q * 255.0, hr_q * 255.0),
        "edge_psnr": compute_edge_psnr(sr_y, hr_y, edge_q),
        "lpips": compute_lpips(sr_q, hr_q, lpips_model),
        "psnr_y_pre_rgb": None if sr_y_norm is None else _pre_rgb_psnr(sr_y_norm, hr_y, border),
    }
    return metrics, sr


def _display_u8(rgb: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) → (H, W, 3) uint8. 자르기·양자화만 하고 가장자리는 자르지 않는다(보기용)."""
    return (quantize_and_shave(rgb, border=0)[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()


def _roi_origin(hr_rgb: torch.Tensor, size: int) -> tuple[int, int]:
    """HR Y Sobel 크기 평균이 가장 큰 `size`×`size` 창의 좌상단 (행, 열). 동률이면 먼저 나온 위치."""
    magnitude = _sobel_magnitude(rgb_to_y(hr_rgb)).unsqueeze(0).unsqueeze(0)
    energy = F.avg_pool2d(magnitude, size, stride=1)[0, 0]
    index = int(torch.argmax(energy.flatten()).item())
    return divmod(index, energy.shape[1])


def _save_representative(path_stem: Path, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> None:
    """Before(bicubic 확대) | After(SR) | HR 가로 배치와, 같은 고주파 관심 영역의 확대본을 PNG로 저장한다."""
    with torch.inference_mode():
        before = bicubic_resize(lr, float(scale))
    panels = [_display_u8(before), _display_u8(sr), _display_u8(hr)]
    Image.fromarray(np.concatenate(panels, axis=1)).save(path_stem.with_suffix(".png"))
    height, width = hr.shape[-2:]
    size = min(ROI_SIZE_HR, height, width)
    top, left = _roi_origin(hr, size)
    # 확대본은 보기용 최근접 확대(화소 반복)다. 지표 계산에는 쓰이지 않아 common.bicubic 규칙 대상이 아니다.
    crops = [np.repeat(np.repeat(p[top : top + size, left : left + size], ROI_ZOOM, 0), ROI_ZOOM, 1) for p in panels]
    Image.fromarray(np.concatenate(crops, axis=1)).save(path_stem.parent / f"{path_stem.name}_roi.png")


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_model(
    model: nn.Module | None,
    run_meta: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    *,
    fair_comparison: dict[str, Any],
    device: str | torch.device = "cpu",
    lpips_model: torch.nn.Module | None = None,
) -> tuple[list[SummaryRecord], list[PerImageRecord]]:
    """테스트셋 4종에 대해 평가를 수행하고 summary/per_image 기록 및 대표 영상 이미지를 저장합니다.

    `model=None`은 Bicubic 하한(학습 없는 구성)이며 같은 경로를 지난다. 모든 구성의 SR 결과는
    [0, 1] 자르기 → 8비트 양자화 → 가장자리 자르기를 거쳐 PSNR(Y)·SSIM(Y)·PSNR(RGB)·에지 PSNR·LPIPS로 잰다.
    B 계열(`forward_raw`가 `sr_y_norm`을 내는 모델)은 `psnr_y_pre_rgb`도 기록한다. 공정 비교 조건
    (`shave_border`, `datasets`, `edge_top_q`, LPIPS 백본, 추론 시간 조건, 대표 영상)은 `fair_comparison`에서만 읽는다.

    산출물(`output_dir`): `summary.csv`, `per_image.csv`, 대표 영상이 지정돼 있으면
    `images/<데이터셋>/<image_id>_{median,worst}{,_roi}.png`. 대표 영상 id가 `null`이면 이미지만 건너뛴다.
    이미 같은 이름의 기록이 있으면 `FileExistsError`로 거부하고, 임시 폴더에 다 쓴 뒤 옮긴다(옮기다 실패하면
    옮긴 것을 지운다). 모델은 평가 동안만 평가 모드로 두고 끝나면 호출자의 학습/평가 모드로 되돌린다.

    Raises:
        ValueError: `run_meta`·공정 비교 조건·대표 영상 id가 잘못됐을 때(아무것도 쓰기 전에), 또는
            `run_meta['switches']`의 B와 모델의 `sr_y_norm` 출력 여부가 다를 때(기록을 남기지 않음).
        FileExistsError: 같은 이름의 기록이 이미 있을 때.
    """
    border, scale, datasets, edge_q, bench = _check_inputs(run_meta, fair_comparison)
    output_dir = Path(output_dir)
    existing = [name for name in (SUMMARY_FILENAME, PER_IMAGE_FILENAME, IMAGES_DIRNAME) if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"{output_dir}에 이전 평가 기록이 있어 덮어쓰지 않습니다: {existing}")
    target = torch.device(device)
    switches = _switches_label(run_meta["switches"])
    representative = fair_comparison["pilot_determined"]["representative_images"]

    loaded = {name: TestsetDataset(data_dir, name) for name in datasets}
    for name, dataset in loaded.items():
        if dataset.scale != scale:
            raise ValueError(f"테스트셋 {name}의 배율 {dataset.scale}이 공정 비교 조건의 배율 {scale}과 다릅니다.")
        known = set(dataset.image_ids)
        for kind, image_id in representative.get(name, {}).items():
            if image_id is not None and image_id not in known:
                raise ValueError(f"대표 영상 {name}.{kind}={image_id!r}이(가) 테스트셋에 없습니다.")

    with _evaluation_mode(model, target):
        return _evaluate_and_write(
            model,
            run_meta,
            loaded,
            output_dir,
            switches=switches,
            scale=scale,
            border=border,
            edge_q=edge_q,
            bench=bench,
            representative=representative,
            lpips_model=lpips_model,
            target=target,
        )


@contextmanager
def _evaluation_mode(model: nn.Module | None, device: torch.device) -> Iterator[None]:
    """평가 동안만 모델을 `device`·평가 모드로 두고, 끝나면(예외가 나도) 호출자의 학습/평가 모드를 되돌린다."""
    if model is None:
        yield
        return
    was_training = model.training
    model.to(device).eval()
    try:
        yield
    finally:
        model.train(was_training)


def _move_from_staging(staging: Path, output_dir: Path) -> None:
    """임시 폴더의 산출물을 `output_dir`로 옮긴다. 도중에 실패하면 이미 옮긴 것을 지워 반쪽 기록을 남기지 않는다.

    옮길 이름들은 호출 전에 `output_dir`에 없음을 확인했으므로, 지워도 이전 기록을 잃지 않는다.
    """
    moved: list[Path] = []
    try:
        for entry in sorted(staging.iterdir()):
            destination = output_dir / entry.name
            os.replace(entry, destination)
            moved.append(destination)
    except BaseException:
        for destination in moved:
            if destination.is_dir():
                shutil.rmtree(destination, ignore_errors=True)
            else:
                destination.unlink(missing_ok=True)
        raise


def _evaluate_and_write(
    model: nn.Module | None,
    run_meta: dict[str, Any],
    loaded: dict[str, TestsetDataset],
    output_dir: Path,
    *,
    switches: str,
    scale: int,
    border: int,
    edge_q: float,
    bench: dict[str, Any],
    representative: dict[str, dict[str, str | None]],
    lpips_model: torch.nn.Module | None,
    target: torch.device,
) -> tuple[list[SummaryRecord], list[PerImageRecord]]:
    """추론 시간을 재고, 테스트셋마다 영상별 지표를 계산해 임시 폴더에 쓴 뒤 `output_dir`로 옮긴다."""
    switch_b = bool(run_meta["switches"].get("B", False))
    infer_ms_net, infer_ms_total = measure_inference_time(
        model,
        scale=scale,
        lr_height=bench["lr_height"],
        lr_width=bench["lr_width"],
        warmup_runs=bench["warmup_runs"],
        repeat_runs=bench["repeat_runs"],
        device=target,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / _STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        summaries: list[SummaryRecord] = []
        per_images: list[PerImageRecord] = []
        for name, dataset in loaded.items():
            rows: list[dict[str, float | None]] = []
            for index in range(len(dataset)):
                sample = dataset[index]
                metrics, sr = _evaluate_image(
                    model,
                    sample,
                    scale=scale,
                    border=border,
                    edge_q=edge_q,
                    switch_b=switch_b,
                    lpips_model=lpips_model,
                    device=target,
                )
                rows.append(metrics)
                per_images.append(
                    PerImageRecord(
                        run_id=run_meta["run_id"],
                        dataset=name,
                        image_id=sample["image_id"],
                        **{key: metrics[key] for key in _METRIC_KEYS},
                    )
                )
                for kind, image_id in representative.get(name, {}).items():
                    if image_id == sample["image_id"]:
                        directory = staging / IMAGES_DIRNAME / name
                        directory.mkdir(parents=True, exist_ok=True)
                        _save_representative(
                            directory / f"{image_id}_{kind}", sample["lr"].to(target), sr, sample["hr"].to(target), scale
                        )
            pre_rgb = [r["psnr_y_pre_rgb"] for r in rows if r["psnr_y_pre_rgb"] is not None]
            summaries.append(
                SummaryRecord(
                    run_id=run_meta["run_id"],
                    commit=run_meta["commit"],
                    config_id=run_meta["config_id"],
                    seed=int(run_meta["seed"]),
                    patch_pool_id=run_meta["patch_pool_id"],
                    switches=switches,
                    dataset=name,
                    **{key: _mean([r[key] for r in rows]) for key in _METRIC_KEYS},
                    psnr_y_pre_rgb=_mean(pre_rgb) if pre_rgb else None,
                    params=int(run_meta["params"]),
                    infer_ms_net=infer_ms_net,
                    infer_ms_total=infer_ms_total,
                    train_minutes=float(run_meta["train_minutes"]),
                    status=run_meta["status"],
                )
            )
        pd.DataFrame([asdict(r) for r in per_images]).to_csv(staging / PER_IMAGE_FILENAME, index=False)
        pd.DataFrame([asdict(r) for r in summaries]).to_csv(staging / SUMMARY_FILENAME, index=False)
        _move_from_staging(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return summaries, per_images
