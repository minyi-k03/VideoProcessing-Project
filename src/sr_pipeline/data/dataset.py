"""패치 풀 데이터셋(학습)과 테스트셋 데이터셋(평가) 로더."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sr_pipeline.data.prepare import METADATA_FILENAME, POOL_DIRNAME, TESTSETS_DIRNAME, _list_images
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
    """Set5, Set14, BSD100, Urban100 평가용 (LR, HR, image_id) 데이터셋.

    `data.prepare.prepare_testsets`의 산출물 `<data_dir>/testsets/<데이터셋>/{LR,HR}/<image_id>.png`를 읽는다.
    평가는 modcrop을 거쳐 저장된 HR만 쓴다. 영상은 image_id(파일명) 순이며, 각 항목은
    `{"image_id": str, "lr": (1, 3, h, w), "hr": (1, 3, h·배율, w·배율)}`(float32, [0, 1])다.
    """

    __test__ = False  # 이름이 Test로 시작해도 pytest 수집 대상이 아니다.

    def __init__(self, data_dir: Path, dataset_name: str) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        testsets_dir = Path(data_dir) / TESTSETS_DIRNAME
        scale = _read_testset_scale(testsets_dir)
        self.scale = scale
        lr_dir = testsets_dir / dataset_name / "LR"
        hr_dir = testsets_dir / dataset_name / "HR"
        for directory in (lr_dir, hr_dir):
            if not directory.is_dir():
                raise ValueError(f"테스트셋 {dataset_name}의 폴더가 없습니다: {directory}")
        lr_paths = {p.stem: p for p in _list_images(lr_dir)}
        hr_paths = {p.stem: p for p in _list_images(hr_dir)}
        if not hr_paths:
            raise ValueError(f"테스트셋 {dataset_name}에 영상이 없습니다: {hr_dir}")
        if lr_paths.keys() != hr_paths.keys():
            only_lr = sorted(lr_paths.keys() - hr_paths.keys())
            only_hr = sorted(hr_paths.keys() - lr_paths.keys())
            raise ValueError(f"테스트셋 {dataset_name}의 LR·HR 영상 목록이 다릅니다: LR만 {only_lr}, HR만 {only_hr}")
        self._image_ids = sorted(hr_paths)
        self._lr_paths = lr_paths
        self._hr_paths = hr_paths
        for image_id in self._image_ids:  # 크기 대응은 처음에 모두 확인해 평가 도중 실패하지 않게 한다.
            lr_size, hr_size = _image_size(lr_paths[image_id]), _image_size(hr_paths[image_id])
            if hr_size != (lr_size[0] * scale, lr_size[1] * scale):
                raise ValueError(
                    f"{dataset_name}/{image_id}의 HR 크기 {hr_size}가 LR {lr_size}의 x{scale}가 아닙니다."
                )

    @property
    def image_ids(self) -> list[str]:
        """image_id 목록(`__getitem__` 순서). 영상을 읽지 않고 id만 확인할 때 쓴다."""
        return list(self._image_ids)

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(self):
            raise IndexError(f"영상 번호 {idx!r}가 범위 [0, {len(self)})를 벗어났습니다.")
        image_id = self._image_ids[idx]
        return {
            "image_id": image_id,
            "lr": _load_unit_tensor(self._lr_paths[image_id]),
            "hr": _load_unit_tensor(self._hr_paths[image_id]),
        }


def _read_testset_scale(testsets_dir: Path) -> int:
    path = testsets_dir / METADATA_FILENAME
    if not path.is_file():
        raise ValueError(f"테스트셋 metadata.json이 없습니다: {path}")
    scale = json.loads(path.read_text(encoding="utf-8")).get("scale")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(f"테스트셋 metadata.json의 scale은 1 이상의 정수여야 합니다: {scale!r} ({path})")
    return scale


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def _load_unit_tensor(path: Path) -> torch.Tensor:
    """PNG를 (1, 3, H, W) float32 [0, 1] 텐서로 읽는다(uint8 / 255)."""
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
