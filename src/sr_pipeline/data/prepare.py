"""DIV2K 패치 풀 1회 생성 및 테스트셋 LR 생성 (SPEC.md / ADR-0001).

열화 모델은 `common.bicubic.bicubic_resize`(MATLAB 호환, 안티에일리어싱) 하나만 쓴다. PIL은 영상 디코드와
PNG 저장에만 쓰고 resize·색 변환에는 쓰지 않는다. LR의 8비트 반올림은 저장 단계의 양자화다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sr_pipeline.common.bicubic import bicubic_resize

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
IMPLEMENTED_DEGRADATION = "bicubic"  # configs/fair_comparison.yaml fixed_rules.degradation과 같은 값
IMPLEMENTED_BICUBIC_RULE ="matlab_imresize_antialias"  # configs/fair_comparison.yaml fixed_rules.bicubic_rule과 같은 값
POOL_DIRNAME = "patch_pool"
TESTSETS_DIRNAME = "testsets"
METADATA_FILENAME = "metadata.json"


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
    source_image_count: int
    content_sha256: str


def _list_images(directory: Path) -> list[Path]:
    """영상 파일을 파일명 코드포인트 순으로 나열한다.

    디렉터리 순회 순서에 결과가 의존하지 않게 정렬한다. `Path`끼리 비교하면 Windows는 대소문자를 무시하고
    Linux는 구분해 OS마다 순서(곧 패치 풀과 patch_pool_id)가 달라지므로, 파일명 문자열로 정렬한다.
    """
    images = (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images, key=lambda p: p.name)


def _load_rgb(path: Path) -> np.ndarray:
    """영상을 (H, W, 3) uint8 RGB로 디코드한다."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)  # 복사본: 쓰기 가능한 배열이어야 torch로 넘길 수 있다.


def modcrop(hr: np.ndarray, scale: int) -> np.ndarray:
    """(H, W, 3) 영상의 아래·오른쪽을 잘라 높이·너비를 배율의 배수로 만든다."""
    height, width = hr.shape[:2]
    return hr[: height - height % scale, : width - width % scale]


def degrade(hr: np.ndarray, scale: int) -> np.ndarray:
    """(H, W, 3) uint8 HR 영상에 bicubic 열화 모델을 적용해 (H/scale, W/scale, 3) uint8 LR 영상을 만든다."""
    tensor = torch.from_numpy(hr).to(torch.float64).div(255.0).permute(2, 0, 1).unsqueeze(0)
    lr = bicubic_resize(tensor, 1.0 / scale, antialiasing=True)
    lr = lr.squeeze(0).permute(1, 2, 0).mul(255.0).round().clamp(0, 255)
    return lr.to(torch.uint8).numpy()


def _check_degradation_rule(scale: int, degradation: str, bicubic_rule: str) -> None:
    """메타데이터에 적는 열화 조건이 실제로 적용하는 열화와 같은지 확인한다(다르면 기록이 거짓이 된다)."""
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError(f"배율은 1 이상의 정수여야 합니다. 받은 값: {scale!r}")
    if degradation != IMPLEMENTED_DEGRADATION:
        raise ValueError(
            f"구현된 열화 모델은 {IMPLEMENTED_DEGRADATION!r} 하나뿐입니다. 받은 값: {degradation!r}"
        )
    if bicubic_rule != IMPLEMENTED_BICUBIC_RULE:
        raise ValueError(
            f"구현된 bicubic 규칙은 {IMPLEMENTED_BICUBIC_RULE!r} 하나뿐입니다. 받은 값: {bicubic_rule!r}"
        )


def _image_size(path: Path) -> tuple[int, int]:
    """영상 헤더만 읽어 (높이, 너비)를 돌려준다."""
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _check_pool_inputs(
    images: list[Path],
    scale: int,
    patch_size_hr: int,
    num_patches: int,
    degradation: str,
    bicubic_rule: str,
) -> None:
    """패치 풀을 쓰기 전에 조건을 확인한다. 영상 크기는 헤더만 읽어 확인한다."""
    _check_degradation_rule(scale, degradation, bicubic_rule)
    if not images:
        raise ValueError("HR 영상이 없습니다.")
    if num_patches < 1:
        raise ValueError(f"num_patches는 1 이상이어야 합니다: {num_patches}")
    if patch_size_hr < scale or patch_size_hr % scale != 0:
        raise ValueError(f"patch_size_hr({patch_size_hr})는 배율({scale})의 배수여야 합니다.")
    for path in images:
        height, width = _image_size(path)
        if height < patch_size_hr or width < patch_size_hr:
            raise ValueError(
                f"{path.name}({height}x{width})이 패치 크기 {patch_size_hr}보다 작아 패치를 만들 수 없습니다."
            )


def _collect_testset_images(
    testsets_hr_dir: Path, scale: int, dataset_names: Sequence[str]
) -> dict[str, list[Path]]:
    """테스트셋을 쓰기 전에 폴더·영상·이름·크기를 확인하고 데이터셋별 영상 목록을 돌려준다."""
    if not dataset_names:
        raise ValueError("dataset_names가 비어 있습니다.")
    duplicated_names = sorted({name for name in dataset_names if list(dataset_names).count(name) > 1})
    if duplicated_names:
        raise ValueError(f"dataset_names에 중복된 테스트셋이 있습니다: {duplicated_names}")
    images_by_dataset: dict[str, list[Path]] = {}
    for name in dataset_names:
        dataset_dir = testsets_hr_dir / name
        if not dataset_dir.is_dir():
            raise ValueError(f"테스트셋 {name}의 HR 폴더가 없습니다: {dataset_dir}")
        images = _list_images(dataset_dir)
        if not images:
            raise ValueError(f"테스트셋 {name}에 HR 영상이 없습니다: {dataset_dir}")
        stems = [path.stem for path in images]
        duplicated = sorted({stem for stem in stems if stems.count(stem) > 1})
        if duplicated:
            raise ValueError(
                f"테스트셋 {name}에 이름(확장자 제외)이 중복된 영상이 있어 출력 파일을 덮어쓰게 됩니다: {duplicated}"
            )
        for path in images:
            height, width = _image_size(path)
            if height < scale or width < scale:
                raise ValueError(f"테스트셋 {name}의 {path.name}({height}x{width})이 배율 {scale}보다 작습니다.")
        images_by_dataset[name] = images
    return images_by_dataset


def _content_sha256(hr: np.ndarray, lr: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (hr, lr):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array))  # 버퍼를 그대로 넘겨 풀 크기만큼의 바이트 복사를 피한다.
    return digest.hexdigest()


def create_patch_pool(
    div2k_hr_dir: Path,
    output_dir: Path,
    scale: int,
    patch_size_hr: int,
    num_patches: int,
    pool_seed: int,
    degradation: str,
    bicubic_rule: str,
) -> PatchPoolMetadata:
    """DIV2K 학습셋 HR 영상에 bicubic 열화 모델을 적용해 패치 풀과 메타데이터를 생성합니다.

    영상마다 전체 HR을 한 번에 열화해 LR을 만든 뒤, HR 패치와 같은 위치의 LR 영역을 잘라 짝을 짓습니다
    (패치마다 따로 열화하면 패치 가장자리에 반사 패딩 효과가 생긴다). HR 패치의 좌상단은 배율의 배수
    좌표라 LR 패치가 HR 패치와 정확히 대응합니다.

    산출물: `output_dir/patch_pool/{hr.npy, lr.npy, metadata.json}`.
    """
    images = _list_images(div2k_hr_dir)
    _check_pool_inputs(images, scale, patch_size_hr, num_patches, degradation, bicubic_rule)
    pool_dir = output_dir / POOL_DIRNAME
    if pool_dir.exists():
        raise FileExistsError(
            f"{pool_dir}에 이미 패치 풀이 있습니다. 패치 풀은 한 번만 만들어 공유하므로 덮어쓰지 않습니다. "
            "다시 만들려면 폴더를 지우거나 다른 출력 폴더를 쓰세요."
        )
    lr_patch_size = patch_size_hr // scale
    rng = np.random.default_rng(np.random.SeedSequence(pool_seed))

    hr_pool = np.empty((num_patches, patch_size_hr, patch_size_hr, 3), dtype=np.uint8)
    lr_pool = np.empty((num_patches, lr_patch_size, lr_patch_size, 3), dtype=np.uint8)

    base, remainder = divmod(num_patches, len(images))
    filled = 0
    for index, path in enumerate(images):
        count = base + (1 if index < remainder else 0)
        if count == 0:  # num_patches가 영상 수보다 적은 풀: 뽑을 패치가 없으면 디코드·열화를 건너뛴다.
            continue
        hr = _load_rgb(path)
        height, width = hr.shape[:2]
        lr = degrade(modcrop(hr, scale), scale)
        for _ in range(count):
            top = int(rng.integers(0, (height - patch_size_hr) // scale + 1)) * scale
            left = int(rng.integers(0, (width - patch_size_hr) // scale + 1)) * scale
            hr_pool[filled] = hr[top : top + patch_size_hr, left : left + patch_size_hr]
            lr_top, lr_left = top // scale, left // scale
            lr_pool[filled] = lr[lr_top : lr_top + lr_patch_size, lr_left : lr_left + lr_patch_size]
            filled += 1

    sha256 = _content_sha256(hr_pool, lr_pool)
    metadata = PatchPoolMetadata(
        patch_pool_id=f"div2k_x{scale}_p{patch_size_hr}_n{num_patches}_s{pool_seed}_{sha256[:8]}",
        scale=scale,
        degradation=degradation,
        bicubic_rule=bicubic_rule,
        patch_size_hr=patch_size_hr,
        num_patches=num_patches,
        pool_seed=pool_seed,
        source_image_count=len(images),
        content_sha256=sha256,
    )

    pool_dir.mkdir(parents=True)
    np.save(pool_dir / "hr.npy", hr_pool)
    np.save(pool_dir / "lr.npy", lr_pool)
    (pool_dir / METADATA_FILENAME).write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def _check_archive_location(data_dir: Path, archive_path: Path) -> None:
    """압축 파일이 데이터 폴더 안에 있으면 자기 자신을 담게 되므로 거부한다(폴더가 아직 없어도 된다)."""
    data_dir = data_dir.resolve()
    archive_path = archive_path.resolve()
    if archive_path == data_dir or data_dir in archive_path.parents:
        raise ValueError(f"압축 파일({archive_path})은 데이터 폴더({data_dir}) 밖에 만들어야 합니다.")


def pack_archive(data_dir: Path, archive_path: Path) -> Path:
    """데이터 폴더의 패치 풀·테스트셋을 압축 파일 1개로 묶습니다(ADR-0005: 세션마다 이 파일 하나만 복사해 푼다).

    zip 안의 경로는 `data_dir` 기준 상대 경로(`patch_pool/...`, `testsets/...`)입니다. 옆의 임시 파일에
    다 쓴 뒤 `archive_path`로 바꿔, 중간에 실패해도 반쯤 쓴 압축 파일이 그 이름으로 남지 않습니다.
    """
    _check_archive_location(data_dir, archive_path)
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"묶을 파일이 없습니다: {data_dir}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(data_dir).as_posix())
        partial.replace(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return archive_path


def prepare_dataset_archive(
    div2k_hr_dir: Path,
    testsets_hr_dir: Path,
    output_dir: Path,
    archive_path: Path,
    scale: int,
    dataset_names: Sequence[str],
    degradation: str,
    bicubic_rule: str,
    patch_size_hr: int,
    num_patches: int,
    pool_seed: int,
) -> PatchPoolMetadata:
    """테스트셋 LR·패치 풀을 만들고 압축 파일 1개로 묶습니다. 실패하면 아무것도 남기지 않습니다.

    1. 쓰기 전에 모든 조건을 확인합니다: 출력 폴더·압축 파일이 이미 있으면 거부, 압축 파일 위치,
       패치 풀 입력, 테스트셋 입력. 그래서 잘못된 인자는 긴 생성 작업 전에 실패합니다.
    2. `output_dir` 옆의 임시 폴더에서 테스트셋 → 패치 풀 → 압축 순으로 만듭니다.
    3. 성공하면 임시 폴더를 `output_dir`로 옮기고, 실패하면 임시 폴더와 압축 파일을 지운 뒤 예외를 다시 던집니다.
    """
    for path in (output_dir, archive_path):
        if path.exists():
            raise FileExistsError(f"{path}가 이미 있습니다. 덮어쓰지 않으니 지우거나 다른 경로를 쓰세요.")
    _check_archive_location(output_dir, archive_path)
    _check_pool_inputs(
        _list_images(div2k_hr_dir), scale, patch_size_hr, num_patches, degradation, bicubic_rule
    )
    _collect_testset_images(testsets_hr_dir, scale, dataset_names)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.partial-", dir=output_dir.parent))
    archive_written = False
    try:
        prepare_testsets(testsets_hr_dir, staging, scale, dataset_names, degradation, bicubic_rule)
        metadata = create_patch_pool(
            div2k_hr_dir, staging, scale, patch_size_hr, num_patches, pool_seed, degradation, bicubic_rule
        )
        pack_archive(staging, archive_path)
        archive_written = True
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if archive_written:
            archive_path.unlink(missing_ok=True)
        raise
    return metadata


def prepare_testsets(
    testsets_hr_dir: Path,
    output_dir: Path,
    scale: int,
    dataset_names: Sequence[str],
    degradation: str,
    bicubic_rule: str,
) -> None:
    """테스트셋 HR 영상에서 패치 풀과 같은 열화 모델로 LR 영상을 생성합니다.

    HR 영상은 배율의 배수가 되도록 오른쪽·아래쪽을 잘라(modcrop) LR이 HR의 정확히 1/배율이 되게 하고,
    잘린 HR도 함께 저장합니다. 평가는 저장된 HR만 씁니다. LR·HR은 무손실 PNG입니다.

    입력: `testsets_hr_dir/<데이터셋>/<영상>`. 산출물: `output_dir/testsets/<데이터셋>/{HR,LR}/<image_id>.png`와
    `output_dir/testsets/metadata.json`.
    """
    _check_degradation_rule(scale, degradation, bicubic_rule)
    images_by_dataset = _collect_testset_images(testsets_hr_dir, scale, dataset_names)

    testsets_dir = output_dir / TESTSETS_DIRNAME
    if testsets_dir.exists():
        raise FileExistsError(
            f"{testsets_dir}에 이미 테스트셋이 있습니다. 이전 실행의 파일이 섞이지 않도록 덮어쓰지 않습니다. "
            "다시 만들려면 폴더를 지우거나 다른 출력 폴더를 쓰세요."
        )
    for name, images in images_by_dataset.items():
        (testsets_dir / name / "HR").mkdir(parents=True)
        (testsets_dir / name / "LR").mkdir(parents=True)
        for path in images:
            hr = modcrop(_load_rgb(path), scale)
            Image.fromarray(hr).save(testsets_dir / name / "HR" / f"{path.stem}.png")
            Image.fromarray(degrade(hr, scale)).save(testsets_dir / name / "LR" / f"{path.stem}.png")

    metadata = {
        "scale": scale,
        "degradation": degradation,
        "bicubic_rule": bicubic_rule,
        "image_counts": {name: len(images) for name, images in images_by_dataset.items()},
    }
    (testsets_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
