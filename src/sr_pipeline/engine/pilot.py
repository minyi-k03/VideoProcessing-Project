"""파일럿 베이스라인 최소 학습·SR 결과 생성 (SPEC.md US51, Task T2.5).

정식 파이프라인(T3.2 구성 해석, T3.3 샘플러, T3.7 트레이너)이 끝나기 전에 진단 입력을 만들기 위한
**파일럿 전용** 경로다. 본 실행에는 쓰이지 않으며 T3.x가 구현되면 대체된다.

- 모델은 `ESPCN()`(모든 스위치가 꺼진 구성 = 베이스라인), 손실은 `SRLoss()`(자르지 않은 SR 결과의 L1)다.
- 옵티마이저는 Adam, 학습률 스케줄은 없다. 이 값들은 파일럿 전용이며 `pilot_determined`에 고정되는 값이 아니다.
- 에폭 수·에폭당 패치 수·배치 크기·학습률은 `pilot_determined`가 `null`인 동안 코드에 기본값을 두지 않고
  호출자가 넘긴다. 쓴 값은 반환값에 남는다.
- 데이터 난수(패치 선택·순서·증강)는 `create_data_rng`에서만 뽑고, 모델 난수는 `set_model_seed`로 고정한다.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sr_pipeline.common.seed import create_data_rng, set_model_seed
from sr_pipeline.evaluation.postprocess import quantize_and_shave
from sr_pipeline.losses.sr_loss import SRLoss
from sr_pipeline.models.espcn import ESPCN

POOL_DIRNAME = "patch_pool"
POOL_METADATA_FILENAME = "metadata.json"
LR_DIRNAME = "LR"
ROTATION_COUNT = 4
AUGMENTATION_COUNT = ROTATION_COUNT * 2  # 회전 4 × 좌우 뒤집기 2


def _check_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}은(는) 1 이상의 정수여야 합니다: {value!r}")


def _check_training_settings(epochs: int, patches_per_epoch: int, batch_size: int, learning_rate: float) -> None:
    _check_positive_int(epochs, "epochs")
    _check_positive_int(patches_per_epoch, "patches_per_epoch")
    _check_positive_int(batch_size, "batch_size")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError(f"learning_rate는 숫자여야 합니다: {learning_rate!r}")
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError(f"learning_rate는 0보다 큰 유한한 수여야 합니다: {learning_rate!r}")


def _load_pool(data_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    pool_dir = Path(data_dir) / POOL_DIRNAME
    metadata = json.loads((pool_dir / POOL_METADATA_FILENAME).read_text(encoding="utf-8"))
    hr = np.load(pool_dir / "hr.npy")
    lr = np.load(pool_dir / "lr.npy")
    if hr.shape[0] != lr.shape[0]:
        raise ValueError(f"패치 풀의 HR·LR 개수가 다릅니다: {hr.shape[0]} vs {lr.shape[0]}")
    return hr, lr, str(metadata["patch_pool_id"])


def _augment(array: np.ndarray, code: int) -> np.ndarray:
    """(H, W, 3) 배열에 8방향 중 하나(코드 0~7)를 적용한다. HR·LR에 같은 코드를 쓴다."""
    rotated = np.rot90(array, code % ROTATION_COUNT, axes=(0, 1))
    return np.ascontiguousarray(rotated[:, ::-1] if code >= ROTATION_COUNT else rotated)


def _to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) uint8을 [0, 1] float32 (B, 3, H, W)로 바꾼다."""
    return torch.from_numpy(batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)


def train_pilot_baseline(
    data_dir: Path,
    checkpoint_path: Path,
    *,
    seed: int,
    epochs: int,
    patches_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> dict[str, Any]:
    """패치 풀로 베이스라인 ESPCN을 1시드 학습하고 마지막 에폭 체크포인트를 저장합니다.

    에폭마다 풀에서 `patches_per_epoch`개를 비복원으로 뽑아(풀보다 많으면 풀 전체를 한 번 더 이어 붙인다)
    뒤섞고, 패치마다 증강 코드를 정해 `batch_size`씩 학습한다. 학습 설정이 잘못되면
    체크포인트를 쓰기 전에 `ValueError`로 멈춘다.

    Returns:
        쓴 설정(`seed`, `epochs`, `patches_per_epoch`, `batch_size`, `learning_rate`), `patch_pool_id`,
        에폭별 평균 손실 `epoch_losses`, `train_minutes`. 입력·저장 경로는 담지 않는다(Public 저장소).
    """
    _check_training_settings(epochs, patches_per_epoch, batch_size, learning_rate)
    hr_pool, lr_pool, patch_pool_id = _load_pool(data_dir)

    set_model_seed(seed)
    data_rng = create_data_rng(seed)
    torch_device = torch.device(device)
    model = ESPCN().to(torch_device)
    loss_fn = SRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pool_size = hr_pool.shape[0]
    epoch_losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        chosen = np.concatenate(
            [data_rng.permutation(pool_size) for _ in range(math.ceil(patches_per_epoch / pool_size))]
        )[:patches_per_epoch]
        data_rng.shuffle(chosen)
        codes = data_rng.integers(0, AUGMENTATION_COUNT, size=patches_per_epoch)

        total, count = 0.0, 0
        for start in range(0, patches_per_epoch, batch_size):
            indices, batch_codes = chosen[start : start + batch_size], codes[start : start + batch_size]
            hr = _to_tensor(np.stack([_augment(hr_pool[i], int(c)) for i, c in zip(indices, batch_codes)]), torch_device)
            lr = _to_tensor(np.stack([_augment(lr_pool[i], int(c)) for i, c in zip(indices, batch_codes)]), torch_device)
            loss = loss_fn(model.forward_raw(lr), hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        epoch_losses.append(total / count)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return {
        "seed": seed,
        "epochs": epochs,
        "patches_per_epoch": patches_per_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patch_pool_id": patch_pool_id,
        "epoch_losses": epoch_losses,
        "train_minutes": (time.perf_counter() - started) / 60.0,
    }


def generate_sr_results(
    checkpoint_path: Path,
    testsets_dir: Path,
    sr_dir: Path,
    dataset_names: Sequence[str],
    *,
    device: str,
) -> dict[str, int]:
    """저장된 베이스라인 체크포인트로 테스트셋 LR 영상의 SR 결과 PNG를 만듭니다 (진단 입력).

    입력은 `<testsets_dir>/<데이터셋>/LR/<image_id>.png`, 저장 위치는 `<sr_dir>/<데이터셋>/<image_id>.png`이다.
    저장 전에 `quantize_and_shave(border=0)`로 [0, 1] 자르기와 8비트 양자화만 하고, 가장자리 잘라내기는
    진단 쪽(`shave_border`)이 한다. 모든 데이터셋 폴더를 먼저 확인하므로 하나라도 없으면 아무것도 쓰기 전에
    `ValueError`로 멈춘다.

    Returns:
        데이터셋별로 만든 SR 영상 수.
    """
    testsets_dir, sr_dir = Path(testsets_dir), Path(sr_dir)
    images_by_dataset: dict[str, list[Path]] = {}
    for name in dataset_names:
        lr_dir = testsets_dir / name / LR_DIRNAME
        if not lr_dir.is_dir():
            raise ValueError(f"테스트셋 {name}의 LR 폴더가 없습니다: {lr_dir}")
        images = sorted(lr_dir.glob("*.png"), key=lambda path: path.name)
        if not images:
            raise ValueError(f"테스트셋 {name}에 LR 영상이 없습니다: {lr_dir}")
        images_by_dataset[name] = images

    torch_device = torch.device(device)
    model = ESPCN().to(torch_device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device, weights_only=True))
    model.eval()

    counts: dict[str, int] = {}
    with torch.no_grad():
        for name, images in images_by_dataset.items():
            (sr_dir / name).mkdir(parents=True, exist_ok=True)
            for path in images:
                with Image.open(path) as image:
                    lr = _to_tensor(np.array(image.convert("RGB"), dtype=np.uint8)[None], torch_device)
                sr = quantize_and_shave(model(lr), border=0)
                sr_uint8 = (sr * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(sr_uint8).save(sr_dir / name / path.name)
            counts[name] = len(images)
    return counts
