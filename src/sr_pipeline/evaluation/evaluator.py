"""단일 평가 경로 통합 실행기 (SPEC.md / ADR-0003).

Bicubic 하한과 모든 학습 구성이 동일한 평가 경로를 통과하도록 강제합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn


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


def evaluate_model(
    model: nn.Module | None,
    run_meta: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> tuple[list[SummaryRecord], list[PerImageRecord]]:
    """테스트셋 4종에 대해 평가를 수행하고 summary/per_image 기록 및 대표 영상 이미지를 저장합니다."""
    raise NotImplementedError("TODO: Run unified evaluation path (including Bicubic when model is None).")
