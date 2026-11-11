"""진단 4종 + 최악 영상 목록 생성 (SPEC.md / ADR-0004).

1주차 파일럿 베이스라인 진단과 본 파이프라인 내 진단 분석이 모두 이 함수들을 공유합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def compute_error_map(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> torch.Tensor:
    """SR 결과와 HR 영상의 화소별 오차 지도를 계산합니다."""
    raise NotImplementedError("TODO: Compute absolute/squared spatial error map.")


def compute_edge_flat_psnr(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
    top_q_percent: float,
) -> dict[str, float]:
    """에지 영역과 평탄 영역의 PSNR(Y)을 분리 계산합니다."""
    raise NotImplementedError("TODO: Return {'edge_psnr_y': ..., 'flat_psnr_y': ...}.")


def compute_frequency_spectrum_diff(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
) -> dict[str, Any]:
    """SR 결과와 HR 영상의 2D FFT 주파수 스펙트럼 및 대역별 오차를 비교합니다."""
    raise NotImplementedError("TODO: Compute 2D FFT magnitude spectrum comparison.")


def decompose_ycbcr_error(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> dict[str, float]:
    """BT.601 YCbCr 채널별(Y, Cb, Cr) 오차 및 비중을 분해합니다."""
    raise NotImplementedError("TODO: Decompose MSE/MAE across Y, Cb, Cr channels.")


def rank_worst_images(per_image_csv: Path, top_k: int = 5) -> list[dict[str, Any]]:
    """테스트셋별 최악 PSNR(Y) 영상 목록을 반환합니다."""
    raise NotImplementedError("TODO: Rank worst-performing images per dataset.")


def run_diagnosis_suite(
    sr_dir: Path,
    hr_dir: Path,
    output_dir: Path,
    top_q_percent: float = 20.0,
) -> None:
    """진단 4종 그림/수치 및 최악·중앙값 영상 요약을 생성합니다."""
    raise NotImplementedError("TODO: Execute full diagnostic suite and save figures/tables.")
