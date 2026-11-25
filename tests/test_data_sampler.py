"""시드 기반 샘플러·패치 풀 데이터셋 바깥 동작 테스트 (SPEC.md US32·33, Task T3.3).

작은 합성 패치 풀만 쓰므로 로컬 CPU에서 돈다. 여기 쓰인 에폭당 패치 수·증강 값은 테스트용이며,
`pilot_determined`가 `null`인 동안 코드에는 기본값을 두지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sr_pipeline.data.dataset import PatchPoolDataset, read_patch_pool_id
from sr_pipeline.data.sampler import SeededEpochPatchSampler

SCALE = 4
POOL_SIZE = 12
PATCHES_PER_EPOCH = 8
POOL_ID = "synthetic_pool"


def _upscale_by_repeat(lr: np.ndarray) -> np.ndarray:
    """(…, h, w, 3) → (…, h·4, w·4, 3) 화소 복제. 회전·뒤집기와 교환되므로 HR·LR 대응의 독립 기준이 된다."""
    return np.repeat(np.repeat(lr, SCALE, axis=-3), SCALE, axis=-2)


def _write_pool(data_dir: Path, lr: np.ndarray, hr: np.ndarray, pool_id: str = POOL_ID) -> None:
    """`data.prepare.create_patch_pool`과 같은 파일 구조의 작은 합성 패치 풀을 쓴다."""
    pool_dir = data_dir / "patch_pool"
    pool_dir.mkdir(parents=True)
    np.save(pool_dir / "hr.npy", hr)
    np.save(pool_dir / "lr.npy", lr)
    metadata = {"patch_pool_id": pool_id, "scale": SCALE, "num_patches": len(lr)}
    (pool_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _random_lr(count: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (count, 4, 4, 3), dtype=np.uint8)


@pytest.fixture
def pool_dir(tmp_path: Path) -> Path:
    lr = _random_lr(POOL_SIZE)
    _write_pool(tmp_path / "data", lr, _upscale_by_repeat(lr))
    return tmp_path / "data"


def _sampler(seed: int = 0, *, pool_size: int = POOL_SIZE, patches: int = PATCHES_PER_EPOCH, augmentation: str = "rot90x4_hflip"):
    return SeededEpochPatchSampler(pool_size, patches, seed, augmentation=augmentation)


def test_epoch_plan_matches_pinned_values_for_seed_0() -> None:
    """난수 소비 순서가 바뀌면 이미 커밋된 파일럿 학습(`results/pilot/`)이 코드로 재현되지 않는다.

    기준 값은 T3.3 이전 파일럿 구현(`engine.pilot.sample_epoch_batches`)에서 한 번 뽑아 고정했다.
    """
    sampler = _sampler(0)

    patch_idx, aug_code = sampler.epoch_plan(0)
    assert patch_idx.tolist() == [6, 4, 10, 2, 8, 7, 11, 3]
    assert aug_code.tolist() == [7, 3, 6, 6, 4, 5, 1, 3]

    patch_idx, aug_code = sampler.epoch_plan(1)
    assert patch_idx.tolist() == [9, 0, 8, 3, 6, 5, 4, 11]
    assert aug_code.tolist() == [4, 5, 4, 0, 2, 1, 3, 5]


def test_no_augmentation_uses_identity_codes_and_draws_no_augmentation_randomness() -> None:
    sampler = _sampler(0, augmentation="none")

    patch_idx, aug_code = sampler.epoch_plan(0)
    assert patch_idx.tolist() == [6, 4, 10, 2, 8, 7, 11, 3]
    assert aug_code.tolist() == [0] * PATCHES_PER_EPOCH
    # 증강 코드를 뽑지 않으므로 다음 에폭의 순서가 증강을 켠 경우와 다르다(파일럿과 같은 기준 값).
    assert sampler.epoch_plan(1)[0].tolist() == [1, 8, 10, 9, 5, 3, 2, 0]


def test_epoch_plans_differ_between_epochs() -> None:
    sampler = _sampler(0)
    assert sampler.epoch_plan(0)[0].tolist() != sampler.epoch_plan(1)[0].tolist()


def test_epoch_can_be_requested_directly_and_matches_sequential_consumption() -> None:
    """재개(T3.7)에서 3번째 에폭부터 시작해도 처음부터 돌린 것과 같은 계획이 나와야 한다."""
    sequential = _sampler(2)
    expected = [sequential.epoch_plan(epoch) for epoch in range(4)]

    fresh = _sampler(2)
    for epoch in (3, 1, 0, 2):  # 순서와 무관해야 한다
        for got, want in zip(fresh.epoch_plan(epoch), expected[epoch]):
            np.testing.assert_array_equal(got, want)


def test_iter_epoch_indices_yields_plan_as_pairs() -> None:
    sampler = _sampler(0)
    patch_idx, aug_code = sampler.epoch_plan(0)

    pairs = list(sampler.iter_epoch_indices(0))

    assert pairs == list(zip(patch_idx.tolist(), aug_code.tolist()))


def test_sampler_does_not_touch_global_random_state() -> None:
    np.random.seed(7)
    expected_next = np.random.rand()
    np.random.seed(7)

    _sampler(0).epoch_plan(2)

    assert np.random.rand() == expected_next


def test_plan_is_independent_of_model_seed_and_torch_random_state() -> None:
    """모델 초기화 난수(구성 스위치로 소비량이 달라질 수 있다)가 데이터 순서를 바꾸면 안 된다."""
    import random

    import torch

    from sr_pipeline.common.seed import set_model_seed

    expected = _sampler(1).epoch_plan(2)

    set_model_seed(99)
    torch.rand(1000)  # 모델 쪽 난수를 많이 소비한 상태
    random.random()
    got = _sampler(1).epoch_plan(2)

    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_epoch_larger_than_pool_covers_pool_evenly() -> None:
    patch_idx, _ = _sampler(0, pool_size=5, patches=13).epoch_plan(0)

    counts = np.bincount(patch_idx, minlength=5)
    assert counts.sum() == 13
    assert set(counts.tolist()) <= {2, 3}  # 풀 2바퀴(10개) + 세 번째 바퀴 3개


def test_epoch_smaller_than_pool_never_repeats_a_patch() -> None:
    patch_idx, _ = _sampler(0, pool_size=12, patches=8).epoch_plan(0)
    assert len(set(patch_idx.tolist())) == 8


def test_augmentation_codes_cover_all_eight_orientations() -> None:
    _, aug_code = _sampler(0, pool_size=64, patches=512).epoch_plan(0)
    assert set(aug_code.tolist()) == set(range(8))


@pytest.mark.parametrize(
    "override",
    [
        {"pool_size": 0},
        {"patches": 0},
        {"patches": True},
        {"pool_size": 2.5},
        {"augmentation": "unknown"},
        {"seed": -1},
    ],
)
def test_sampler_rejects_invalid_settings(override: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        _sampler(**override)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_epoch_plan_rejects_invalid_epoch(epoch) -> None:
    with pytest.raises(ValueError, match="epoch"):
        _sampler(0).epoch_plan(epoch)


def test_same_seed_gives_same_plan_every_time() -> None:
    first, second = _sampler(1), _sampler(1)
    for epoch in range(3):
        for a, b in zip(first.epoch_plan(epoch), second.epoch_plan(epoch)):
            np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_plans() -> None:
    plans = [_sampler(seed).epoch_plan(0)[0].tolist() for seed in (0, 1, 2)]
    assert len({tuple(plan) for plan in plans}) == 3


def test_dataset_reads_pool_and_exposes_id_and_size(pool_dir: Path) -> None:
    dataset = PatchPoolDataset(pool_dir, POOL_ID)
    assert (len(dataset), dataset.patch_pool_id) == (POOL_SIZE, POOL_ID)
    assert read_patch_pool_id(pool_dir) == POOL_ID


def test_dataset_rejects_mismatched_patch_pool_id(pool_dir: Path) -> None:
    with pytest.raises(ValueError, match="patch_pool_id"):
        PatchPoolDataset(pool_dir, "another_pool")


def test_dataset_rejects_pool_with_wrong_hr_lr_scale(tmp_path: Path) -> None:
    lr = _random_lr(3)
    _write_pool(tmp_path / "data", lr, _upscale_by_repeat(lr)[:, :-4])
    with pytest.raises(ValueError, match="x4"):
        PatchPoolDataset(tmp_path / "data", POOL_ID)


def test_epoch_batches_follow_the_sampler_plan_without_augmentation(pool_dir: Path) -> None:
    dataset = PatchPoolDataset(pool_dir, POOL_ID)
    sampler = _sampler(0, augmentation="none")
    patch_idx, _ = sampler.epoch_plan(1)
    lr_pool = np.load(pool_dir / "patch_pool" / "lr.npy")

    batches = list(dataset.iter_epoch_batches(sampler, 1, batch_size=3))

    assert [len(lr) for lr, _ in batches] == [3, 3, 2]
    hr_pool = np.load(pool_dir / "patch_pool" / "hr.npy")
    np.testing.assert_array_equal(np.concatenate([lr for lr, _ in batches]), lr_pool[patch_idx])
    np.testing.assert_array_equal(np.concatenate([hr for _, hr in batches]), hr_pool[patch_idx])


def test_epoch_batches_keep_hr_lr_spatial_correspondence_under_augmentation(tmp_path: Path) -> None:
    lr = _random_lr(6)
    _write_pool(tmp_path / "data", lr, _upscale_by_repeat(lr))
    dataset = PatchPoolDataset(tmp_path / "data", POOL_ID)
    sampler = _sampler(0, pool_size=6, patches=40)

    batches = list(dataset.iter_epoch_batches(sampler, 0, batch_size=8))

    for lr_batch, hr_batch in batches:
        np.testing.assert_array_equal(hr_batch, _upscale_by_repeat(lr_batch))
    originals = {patch.tobytes() for patch in lr}
    augmented = [patch for lr_batch, _ in batches for patch in lr_batch if patch.tobytes() not in originals]
    assert augmented, "증강(회전·뒤집기)이 한 번도 적용되지 않았다"


def test_epoch_batches_apply_each_orientation_code_to_its_patch(tmp_path: Path) -> None:
    """코드 c인 패치는 정확히 (c % 4)번 90° 회전 후 c ≥ 4이면 좌우 뒤집기한 모양이어야 한다."""
    lr = _random_lr(POOL_SIZE)
    _write_pool(tmp_path / "data", lr, _upscale_by_repeat(lr))
    dataset = PatchPoolDataset(tmp_path / "data", POOL_ID)
    sampler = _sampler(0, patches=64)
    patch_idx, aug_code = sampler.epoch_plan(0)

    lr_batches = [lr_batch for lr_batch, _ in dataset.iter_epoch_batches(sampler, 0, batch_size=64)]

    for got, index, code in zip(np.concatenate(lr_batches), patch_idx, aug_code):
        want = lr[index]
        for _ in range(code % 4):
            want = np.rot90(want, 1, axes=(0, 1))  # 반시계 90°
        if code >= 4:
            want = want[:, ::-1]
        np.testing.assert_array_equal(got, want)


def test_epoch_batches_reject_sampler_for_a_different_pool_size(pool_dir: Path) -> None:
    dataset = PatchPoolDataset(pool_dir, POOL_ID)
    with pytest.raises(ValueError, match="pool_size"):
        list(dataset.iter_epoch_batches(_sampler(0, pool_size=POOL_SIZE + 1), 0, batch_size=4))


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_epoch_batches_reject_invalid_batch_size(pool_dir: Path, batch_size) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(PatchPoolDataset(pool_dir, POOL_ID).iter_epoch_batches(_sampler(0), 0, batch_size=batch_size))
