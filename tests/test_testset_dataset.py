"""TestsetDataset 경계 테스트: prepare_testsets 산출물을 평가용 (LR, HR, image_id)로 읽는다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sr_pipeline.data.dataset import TestsetDataset
from sr_pipeline.data.prepare import prepare_testsets


def make_testsets(root: Path, names=("Set5",), sizes=((22, 30), (16, 16))) -> Path:
    """합성 HR(배율의 배수가 아닌 크기 포함)로 `prepare_testsets`를 실행해 data_dir를 돌려준다."""
    rng = np.random.default_rng(0)
    hr_root = root / "hr_src"
    for name in names:
        (hr_root / name).mkdir(parents=True)
        for index, (h, w) in enumerate(sizes):
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(hr_root / name / f"img{index}.png")
    data_dir = root / "data"
    prepare_testsets(hr_root, data_dir, 4, list(names), "bicubic", "matlab_imresize_antialias")
    return data_dir


def test_testset_yields_float_lr_hr_pairs_in_sorted_image_id_order(tmp_path: Path) -> None:
    data_dir = make_testsets(tmp_path)
    dataset = TestsetDataset(data_dir, "Set5")

    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert (first["image_id"], second["image_id"]) == ("img0", "img1")
    # img0: 22x30 -> modcrop 20x28, LR 5x7
    assert first["hr"].shape == (1, 3, 20, 28) and first["lr"].shape == (1, 3, 5, 7)
    assert first["lr"].dtype == torch.float32 and first["hr"].dtype == torch.float32
    assert 0.0 <= first["lr"].min().item() and first["hr"].max().item() <= 1.0
    # 저장된 PNG 값을 /255로 읽는다.
    stored = np.array(Image.open(data_dir / "testsets" / "Set5" / "HR" / "img1.png"))
    assert torch.equal(second["hr"][0], torch.from_numpy(stored).permute(2, 0, 1).float() / 255.0)


def test_testset_rejects_missing_dataset_and_mismatched_lr_hr(tmp_path: Path) -> None:
    data_dir = make_testsets(tmp_path)
    with pytest.raises(ValueError):
        TestsetDataset(data_dir, "Set14")  # 없는 데이터셋

    (data_dir / "testsets" / "Set5" / "LR" / "img1.png").unlink()
    with pytest.raises(ValueError):
        TestsetDataset(data_dir, "Set5")  # LR·HR 영상 id 불일치


def test_testset_rejects_lr_hr_size_not_matching_scale(tmp_path: Path) -> None:
    data_dir = make_testsets(tmp_path)
    Image.fromarray(np.zeros((9, 9, 3), dtype=np.uint8)).save(data_dir / "testsets" / "Set5" / "LR" / "img0.png")
    with pytest.raises(ValueError):
        TestsetDataset(data_dir, "Set5")


def test_testset_index_out_of_range_raises(tmp_path: Path) -> None:
    dataset = TestsetDataset(make_testsets(tmp_path), "Set5")
    with pytest.raises(IndexError):
        dataset[2]
