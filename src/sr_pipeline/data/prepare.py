"""DIV2K 패치 풀 1회 생성 및 테스트셋 LR 생성 (SPEC.md / ADR-0001)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchPoolMetadata:
    """패치 풀 생성 조건 및 식별자 메타데이터."""

    patch_pool_id: str
    scale: int
    degradation: str
    bicubic_rule: str
    patch_size_hr: int
    num_patches: int
    pool_seed: int


def create_patch_pool(
    div2k_hr_dir: Path,
    output_dir: Path,
    scale: int = 4,
    patch_size_hr: int = 96,
    num_patches: int = 10000,
    pool_seed: int = 42,
) -> PatchPoolMetadata:
    """DIV2K 학습셋 HR 영상 800장에 bicubic 열화 모델을 적용해 패치 풀과 메타데이터를 생성합니다."""
    raise NotImplementedError("TODO: Generate shared patch pool once and write metadata JSON.")


def prepare_testsets(
    testsets_hr_dir: Path,
    output_dir: Path,
    scale: int = 4,
) -> None:
    """Set5, Set14, BSD100, Urban100 HR 영상에서 동일 열화 모델로 LR 영상을 생성합니다."""
    raise NotImplementedError("TODO: Generate testset LR images with identical bicubic degradation.")
