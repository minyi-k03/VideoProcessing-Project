"""에폭별 패치 순서·증강을 결정하는 시드 고정 샘플러 (SPEC.md / ADR-0001).

샘플러는 시드에서 파생한 데이터 난수(`common.seed.create_data_rng`)에서만 난수를 뽑는다. 모델 쪽 난수나
구성(스위치)에는 의존하지 않으므로, 같은 시드면 구성이 달라도 같은 패치 순서·증강이 나온다.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from sr_pipeline.common.seed import create_data_rng

ROTATION_COUNT = 4
AUGMENTATION_COUNT = ROTATION_COUNT * 2  # 회전 4 × 좌우 뒤집기 2
AUGMENTATIONS = ("rot90x4_hflip", "none")


def _check_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}은(는) 1 이상의 정수여야 합니다: {value!r}")


def check_augmentation(augmentation: object) -> None:
    """지원하는 증강 이름(`AUGMENTATIONS`)이 아니면 `ValueError`."""
    if not isinstance(augmentation, str) or augmentation not in AUGMENTATIONS:
        raise ValueError(f"augmentation은(는) {AUGMENTATIONS} 중 하나여야 합니다: {augmentation!r}")


def augment_patch(array: np.ndarray, code: int) -> np.ndarray:
    """(H, W, C) 배열에 8방향 중 하나(코드 0~7: 90° 회전 `code % 4`번, 4 이상이면 이어서 좌우 뒤집기)를 적용한다.

    HR·LR 패치 쌍에는 같은 코드를 써야 둘의 공간 대응이 유지된다.

    Raises:
        ValueError: `code`가 0~7의 정수가 아닐 때.
    """
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or not 0 <= code < AUGMENTATION_COUNT:
        raise ValueError(f"증강 code는 0~{AUGMENTATION_COUNT - 1}의 정수여야 합니다: {code!r}")
    rotated = np.rot90(array, code % ROTATION_COUNT, axes=(0, 1))
    return np.ascontiguousarray(rotated[:, ::-1] if code >= ROTATION_COUNT else rotated)


class SeededEpochPatchSampler:
    """같은 시드면 구성과 무관하게 동일한 패치 순서와 증강 시퀀스를 생성합니다."""

    def __init__(self, pool_size: int, patches_per_epoch: int, seed: int, *, augmentation: str) -> None:
        _check_positive_int(pool_size, "pool_size")
        _check_positive_int(patches_per_epoch, "patches_per_epoch")
        check_augmentation(augmentation)
        create_data_rng(seed)  # 시드 검증을 겸한다(정수·0 이상).
        self.pool_size = pool_size
        self.patches_per_epoch = patches_per_epoch
        self.seed = seed
        self.augmentation = augmentation

    def _draw_epoch(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        # 난수 소비 순서(풀 순열 → 뒤섞기 → 증강 코드)는 파일럿 학습(T2.5)과 같다. 바꾸면 이전 파일럿이 재현되지 않는다.
        chosen = np.concatenate(
            [rng.permutation(self.pool_size) for _ in range(math.ceil(self.patches_per_epoch / self.pool_size))]
        )[: self.patches_per_epoch]
        rng.shuffle(chosen)
        if self.augmentation == "none":
            return chosen, np.zeros(self.patches_per_epoch, dtype=np.int64)
        return chosen, rng.integers(0, AUGMENTATION_COUNT, size=self.patches_per_epoch)

    def epoch_plan(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """해당 에폭의 패치 인덱스 순서와 증강 코드(`(patch_idx, aug_code)`)를 반환합니다."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch은(는) 0 이상의 정수여야 합니다: {epoch!r}")
        rng = create_data_rng(self.seed)
        for _ in range(epoch):  # 앞 에폭의 난수를 소비해 넘어간다: 어느 에폭이든 처음부터 순차 소비한 것과 같다.
            self._draw_epoch(rng)
        return self._draw_epoch(rng)

    def iter_epoch_indices(self, epoch: int) -> Iterator[tuple[int, int]]:
        """해당 에폭의 `(patch_idx, aug_code)` 시퀀스를 반환합니다."""
        patch_idx, aug_code = self.epoch_plan(epoch)
        return zip(patch_idx.tolist(), aug_code.tolist())
