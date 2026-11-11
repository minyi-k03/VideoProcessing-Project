"""에폭별 패치 순서·증강을 결정하는 시드 고정 샘플러 (SPEC.md / ADR-0001)."""

from __future__ import annotations

from typing import Iterator


class SeededEpochPatchSampler:
    """같은 시드면 구성과 무관하게 동일한 패치 순서와 증강 시퀀스를 생성합니다."""

    def __init__(self, pool_size: int, patches_per_epoch: int, seed: int) -> None:
        self.pool_size = pool_size
        self.patches_per_epoch = patches_per_epoch
        self.seed = seed

    def iter_epoch_indices(self, epoch: int) -> Iterator[tuple[int, int]]:
        """해당 에폭의 `(patch_idx, aug_code)` 시퀀스를 반환합니다."""
        raise NotImplementedError("TODO: Yield deterministic (patch_idx, aug_code) per epoch from seed.")
