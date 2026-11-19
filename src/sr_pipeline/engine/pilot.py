"""파일럿 베이스라인 최소 학습·SR 결과 생성 (SPEC.md US51, Task T2.5).

정식 파이프라인(T3.2 구성 해석, T3.3 샘플러, T3.7 트레이너)이 끝나기 전에 진단 입력을 만들기 위한
**파일럿 전용** 경로다. 본 실행에는 쓰이지 않으며 T3.x가 구현되면 대체된다.

- 모델은 `ESPCN()`(모든 스위치가 꺼진 구성 = 베이스라인), 손실은 `SRLoss()`(자르지 않은 SR 결과의 L1)다.
- 옵티마이저는 Adam, 학습률 스케줄은 없고, 증강은 8방향(90° 회전 4 × 좌우 뒤집기 2)이다. 이 값들은 파일럿 전용이며
  `pilot_determined`에 고정되는 값이 아니다. 진단 근거를 재현할 수 있도록 학습 기록에 그대로 남긴다.
- 에폭 수·에폭당 패치 수·배치 크기·학습률은 `pilot_determined`가 `null`인 동안 코드에 기본값을 두지 않고
  호출자가 넘긴다. 쓴 값은 반환값에 남는다.
- 데이터 난수(패치 선택·순서·증강)는 `create_data_rng`에서만 뽑고, 모델 난수는 `set_model_seed`로 고정한다.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sr_pipeline.common.seed import create_data_rng, set_model_seed
from sr_pipeline.evaluation.postprocess import quantize_and_shave
from sr_pipeline.losses.sr_loss import SRLoss
from sr_pipeline.models.espcn import ESPCN, SCALE

POOL_DIRNAME = "patch_pool"
POOL_METADATA_FILENAME = "metadata.json"
LR_DIRNAME = "LR"
ROTATION_COUNT = 4
AUGMENTATION_COUNT = ROTATION_COUNT * 2  # 회전 4 × 좌우 뒤집기 2
# 파일럿에서 코드에 고정한 학습 조건. `pilot_determined`의 확정값이 아니며 기록에 그대로 남긴다.
PILOT_OPTIMIZER = "adam"
PILOT_SCHEDULER = "none"
PILOT_AUGMENTATION = "rot90x4_hflip"


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
    if hr.ndim != 4 or lr.ndim != 4 or hr.shape[-1] != 3 or lr.shape[-1] != 3:
        raise ValueError(f"패치 풀은 (N, H, W, 3) 배열이어야 합니다: HR {hr.shape}, LR {lr.shape}")
    if hr.shape[0] != lr.shape[0]:
        raise ValueError(f"패치 풀의 HR·LR 개수가 다릅니다: {hr.shape[0]} vs {lr.shape[0]}")
    if hr.shape[0] == 0:
        raise ValueError("패치 풀이 비어 있습니다.")
    if hr.shape[1:3] != (lr.shape[1] * SCALE, lr.shape[2] * SCALE):
        raise ValueError(f"HR 패치 크기가 LR의 x{SCALE}가 아닙니다: HR {hr.shape[1:3]}, LR {lr.shape[1:3]}")
    return hr, lr, str(metadata["patch_pool_id"])


def _augment(array: np.ndarray, code: int) -> np.ndarray:
    """(H, W, 3) 배열에 8방향 중 하나(코드 0~7)를 적용한다. HR·LR에 같은 코드를 쓴다."""
    rotated = np.rot90(array, code % ROTATION_COUNT, axes=(0, 1))
    return np.ascontiguousarray(rotated[:, ::-1] if code >= ROTATION_COUNT else rotated)


def _to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) uint8을 [0, 1] float32 (B, 3, H, W)로 바꾼다."""
    return torch.from_numpy(batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)


def sample_epoch_batches(
    hr_pool: np.ndarray,
    lr_pool: np.ndarray,
    rng: np.random.Generator,
    patches_per_epoch: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """한 에폭의 (LR, HR) uint8 배치 `(B, h, w, 3)`, `(B, h·배율, w·배율, 3)`을 차례로 내보냅니다.

    풀의 순열을 필요한 만큼 이어 붙여 앞에서 `patches_per_epoch`개를 고르고(풀보다 적으면 비복원,
    많으면 풀을 여러 바퀴 돈다) 뒤섞은 뒤, 패치마다 8방향 증강 코드 하나를 정해 HR·LR에 같은 변환을 적용한다.
    난수는 `rng`에서만 뽑는다. 마지막 배치는 `batch_size`보다 작을 수 있다.
    """
    pool_size = hr_pool.shape[0]
    chosen = np.concatenate(
        [rng.permutation(pool_size) for _ in range(math.ceil(patches_per_epoch / pool_size))]
    )[:patches_per_epoch]
    rng.shuffle(chosen)
    codes = rng.integers(0, AUGMENTATION_COUNT, size=patches_per_epoch)
    for start in range(0, patches_per_epoch, batch_size):
        pairs = list(zip(chosen[start : start + batch_size], codes[start : start + batch_size]))
        lr = np.stack([_augment(lr_pool[index], int(code)) for index, code in pairs])
        hr = np.stack([_augment(hr_pool[index], int(code)) for index, code in pairs])
        yield lr, hr


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

    에폭 배치는 `sample_epoch_batches`가 만든다. 학습 설정·시드·패치 풀이 잘못되면 체크포인트를 쓰기 전에
    `ValueError`로 멈춘다.

    Returns:
        쓴 설정(`seed`, `epochs`, `patches_per_epoch`, `batch_size`, `learning_rate`), 코드에 고정한 조건
        (`optimizer`, `scheduler`, `augmentation`), `patch_pool_id`, 에폭별 평균 손실 `epoch_losses`,
        `train_minutes`. 입력·저장 경로는 담지 않는다(Public 저장소).
    """
    _check_training_settings(epochs, patches_per_epoch, batch_size, learning_rate)
    data_rng = create_data_rng(seed)  # 시드 검증을 겸한다: 잘못된 시드는 대용량 풀을 읽기 전에 실패한다.
    hr_pool, lr_pool, patch_pool_id = _load_pool(data_dir)

    set_model_seed(seed)
    torch_device = torch.device(device)
    model = ESPCN().to(torch_device)
    loss_fn = SRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        loss_sum, patch_count = 0.0, 0
        for lr_batch, hr_batch in sample_epoch_batches(hr_pool, lr_pool, data_rng, patches_per_epoch, batch_size):
            lr, hr = _to_tensor(lr_batch, torch_device), _to_tensor(hr_batch, torch_device)
            loss = loss_fn(model.forward_raw(lr), hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(lr_batch)
            patch_count += len(lr_batch)
        epoch_losses.append(loss_sum / patch_count)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint_path.with_name(checkpoint_path.name + ".partial")
    try:  # 다 쓴 뒤 이름을 바꿔, 저장 중 실패해도 반쯤 쓴 체크포인트가 그 이름으로 남지 않게 한다.
        torch.save(model.state_dict(), partial)
        partial.replace(checkpoint_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "seed": seed,
        "epochs": epochs,
        "patches_per_epoch": patches_per_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": PILOT_OPTIMIZER,
        "scheduler": PILOT_SCHEDULER,
        "augmentation": PILOT_AUGMENTATION,
        "patch_pool_id": patch_pool_id,
        "epoch_losses": epoch_losses,
        "train_minutes": (time.perf_counter() - started) / 60.0,
    }


def resolve_testset_names(fair_comparison: Any) -> list[str]:
    """`fair_comparison.yaml` 내용에서 테스트셋 이름(`fixed_rules.datasets`)을 꺼냅니다.

    Raises:
        ValueError: 파일 구조가 매핑이 아니거나, `datasets`가 비어 있지 않은 문자열 목록이 아닐 때.
    """
    if not isinstance(fair_comparison, dict) or not isinstance(fair_comparison.get("fixed_rules"), dict):
        raise ValueError("fair_comparison.yaml에 fixed_rules 매핑이 없습니다.")
    names = fair_comparison["fixed_rules"].get("datasets")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"fixed_rules.datasets는 비어 있지 않은 문자열 목록이어야 합니다: {names!r}")
    return names


def collect_testset_lr_images(testsets_dir: Path, dataset_names: Sequence[str]) -> dict[str, list[Path]]:
    """테스트셋별 LR 영상 목록(파일명 순)을 돌려줍니다. 학습 전에 불러 입력 오류를 일찍 잡는 데 씁니다.

    Raises:
        ValueError: 데이터셋 이름이 비었거나, `<testsets_dir>/<데이터셋>/LR` 폴더나 그 안의 PNG가 없을 때.
    """
    if not dataset_names:
        raise ValueError("dataset_names가 비어 있습니다.")
    testsets_dir = Path(testsets_dir)
    images_by_dataset: dict[str, list[Path]] = {}
    for name in dataset_names:
        lr_dir = testsets_dir / name / LR_DIRNAME
        if not lr_dir.is_dir():
            raise ValueError(f"테스트셋 {name}의 LR 폴더가 없습니다: {lr_dir}")
        images = sorted(lr_dir.glob("*.png"), key=lambda path: path.name)
        if not images:
            raise ValueError(f"테스트셋 {name}에 LR 영상이 없습니다: {lr_dir}")
        images_by_dataset[name] = images
    return images_by_dataset


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
    진단 쪽(`shave_border`)이 한다.

    `sr_dir`가 이미 있으면 이전 실행의 SR 결과와 섞이지 않도록 거부한다. `sr_dir` 옆 임시 폴더에 다 쓴 뒤
    이름을 바꾸므로, 중간에 실패하면 아무것도 남기지 않는다.

    Returns:
        데이터셋별로 만든 SR 결과 수.
    """
    sr_dir = Path(sr_dir)
    if sr_dir.exists():
        raise FileExistsError(f"{sr_dir}가 이미 있습니다. 이전 SR 결과와 섞이지 않도록 지우거나 다른 경로를 쓰세요.")
    images_by_dataset = collect_testset_lr_images(testsets_dir, dataset_names)

    torch_device = torch.device(device)
    model = ESPCN().to(torch_device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device, weights_only=True))
    model.eval()

    sr_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{sr_dir.name}.partial-", dir=sr_dir.parent))
    counts: dict[str, int] = {}
    try:
        with torch.no_grad():
            for name, images in images_by_dataset.items():
                (staging / name).mkdir()
                for path in images:
                    with Image.open(path) as image:
                        lr = _to_tensor(np.array(image.convert("RGB"), dtype=np.uint8)[None], torch_device)
                    sr = quantize_and_shave(model(lr), border=0)
                    # quantize_and_shave가 이미 8비트 격자로 맞춘 값이라 여기서는 PNG 저장용 정수로만 바꾼다.
                    sr_uint8 = (sr * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(sr_uint8).save(staging / name / path.name)
                counts[name] = len(images)
        staging.rename(sr_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return counts
