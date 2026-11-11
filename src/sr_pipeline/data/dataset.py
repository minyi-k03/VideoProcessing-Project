"""패치 풀 및 테스트셋 데이터셋 로더 스켈레톤."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PatchPoolDataset:
    """공유 패치 풀과 SeededEpochPatchSampler를 연결하는 학습 데이터셋."""

    def __init__(self, data_dir: Path, patch_pool_id: str) -> None:
        self.data_dir = data_dir
        self.patch_pool_id = patch_pool_id
        raise NotImplementedError("TODO: Load shared patch pool and verify patch_pool_id metadata.")


class TestsetDataset:
    """Set5, Set14, BSD100, Urban100 평가용 (LR, HR, image_id) 데이터셋."""

    def __init__(self, data_dir: Path, dataset_name: str) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        raise NotImplementedError("TODO: Load paired LR/HR test images for evaluation.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError
