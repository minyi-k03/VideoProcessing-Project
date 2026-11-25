"""패치 풀 데이터셋 로더. `TestsetDataset`은 T3.6(평가)에서 구현할 스켈레톤이다."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from sr_pipeline.data.prepare import METADATA_FILENAME, POOL_DIRNAME
from sr_pipeline.data.sampler import SeededEpochPatchSampler, augment_patch


def _read_pool_metadata(data_dir: Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / POOL_DIRNAME / METADATA_FILENAME).read_text(encoding="utf-8"))


def read_patch_pool_id(data_dir: Path) -> str:
    """`<data_dir>/patch_pool/metadata.json`의 `patch_pool_id`를 읽습니다(큰 배열은 읽지 않는다)."""
    return str(_read_pool_metadata(data_dir)["patch_pool_id"])


class PatchPoolDataset:
    """공유 패치 풀과 SeededEpochPatchSampler를 연결하는 패치 풀 데이터셋.

    풀은 `<data_dir>/patch_pool/{hr.npy, lr.npy, metadata.json}`(`data.prepare.create_patch_pool`의 산출물)이며,
    HR·LR 패치는 uint8 `(N, H, W, 3)` 배열로 메모리에 올린다. 배율(`scale`)은 풀 메타데이터의 값을 쓰며,
    모델 배율과 맞는지는 호출자가 확인한다.
    """

    def __init__(self, data_dir: Path, patch_pool_id: str) -> None:
        self.data_dir = data_dir
        metadata = _read_pool_metadata(data_dir)
        found_id = str(metadata["patch_pool_id"])
        if found_id != patch_pool_id:
            raise ValueError(f"patch_pool_id가 다릅니다: 요청 {patch_pool_id!r}, 데이터 {found_id!r} ({data_dir})")
        self.patch_pool_id = found_id
        scale = metadata.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"패치 풀 metadata.json의 scale은 1 이상의 정수여야 합니다: {scale!r} ({data_dir})")
        self.scale = scale
        pool_dir = Path(data_dir) / POOL_DIRNAME
        hr = np.load(pool_dir / "hr.npy")
        lr = np.load(pool_dir / "lr.npy")
        if hr.ndim != 4 or lr.ndim != 4 or hr.shape[-1] != 3 or lr.shape[-1] != 3:
            raise ValueError(f"패치 풀은 (N, H, W, 3) 배열이어야 합니다: HR {hr.shape}, LR {lr.shape}")
        if hr.shape[0] != lr.shape[0]:
            raise ValueError(f"패치 풀의 HR·LR 개수가 다릅니다: {hr.shape[0]} vs {lr.shape[0]}")
        if hr.shape[0] == 0:
            raise ValueError("패치 풀이 비어 있습니다.")
        if hr.shape[1:3] != (lr.shape[1] * scale, lr.shape[2] * scale):
            raise ValueError(f"HR 패치 크기가 LR의 x{scale}가 아닙니다: HR {hr.shape[1:3]}, LR {lr.shape[1:3]}")
        self._hr = hr
        self._lr = lr

    def __len__(self) -> int:
        return self._hr.shape[0]

    @property
    def hr_patch_size(self) -> tuple[int, int]:
        """HR 패치의 `(높이, 너비)`. 학습 설정의 `patch_size_hr`과 맞는지는 호출자가 확인한다."""
        return int(self._hr.shape[1]), int(self._hr.shape[2])

    def iter_epoch_batches(
        self, sampler: SeededEpochPatchSampler, epoch: int, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """한 에폭의 (LR, HR) uint8 배치 `(B, h, w, 3)`, `(B, h·배율, w·배율, 3)`을 차례로 내보냅니다.

        패치와 순서, 증강은 `sampler.epoch_plan(epoch)`이 정한다. HR·LR에 같은 증강 코드를 적용한다.
        마지막 배치는 `batch_size`보다 작을 수 있다.

        Raises:
            ValueError: `batch_size`가 1 이상의 정수가 아니거나 `sampler.pool_size`가 풀 크기와 다를 때.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size은(는) 1 이상의 정수여야 합니다: {batch_size!r}")
        if sampler.pool_size != len(self):
            raise ValueError(f"샘플러의 pool_size({sampler.pool_size})가 패치 풀 크기({len(self)})와 다릅니다.")
        patch_idx, aug_code = sampler.epoch_plan(epoch)
        for start in range(0, len(patch_idx), batch_size):
            pairs = list(zip(patch_idx[start : start + batch_size], aug_code[start : start + batch_size]))
            lr = np.stack([augment_patch(self._lr[index], int(code)) for index, code in pairs])
            hr = np.stack([augment_patch(self._hr[index], int(code)) for index, code in pairs])
            yield lr, hr


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
