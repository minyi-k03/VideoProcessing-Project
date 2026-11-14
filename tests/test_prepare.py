"""데이터 준비 경계 테스트: 패치 풀·테스트셋 LR 생성·압축 (SPEC.md Testing Decisions, T1.4)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sr_pipeline.common.bicubic import bicubic_resize
from sr_pipeline.data.prepare import PatchPoolMetadata, create_patch_pool, pack_archive, prepare_testsets

SCALE = 4
RULE = "matlab_imresize_antialias"


def _write_noise_images(directory: Path, sizes: list[tuple[int, int]], seed: int = 0) -> list[Path]:
    """잡음 RGB 영상(높이, 너비)들을 PNG로 쓴다. 잡음이라 영상·패치 위치가 유일하게 식별된다."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for index, (height, width) in enumerate(sizes):
        array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = directory / f"img_{index:03d}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def _make_pool(hr_dir: Path, output_dir: Path, **overrides) -> PatchPoolMetadata:
    arguments = dict(
        scale=SCALE,
        patch_size_hr=16,
        num_patches=10,
        pool_seed=3,
        degradation="bicubic",
        bicubic_rule=RULE,
    )
    arguments.update(overrides)
    return create_patch_pool(hr_dir, output_dir, **arguments)


def test_patch_pool_arrays_and_metadata_record_generation_conditions(tmp_path: Path) -> None:
    """패치 풀은 HR·LR 8비트 배열과, 생성 조건·식별자를 담은 metadata.json을 만든다."""
    hr_dir = tmp_path / "hr"
    _write_noise_images(hr_dir, [(40, 52), (36, 44), (48, 40)])

    metadata = _make_pool(hr_dir, tmp_path / "out")

    pool_dir = tmp_path / "out" / "patch_pool"
    hr = np.load(pool_dir / "hr.npy")
    lr = np.load(pool_dir / "lr.npy")
    assert hr.dtype == np.uint8 and hr.shape == (10, 16, 16, 3)
    assert lr.dtype == np.uint8 and lr.shape == (10, 4, 4, 3)

    assert metadata.scale == SCALE
    assert metadata.degradation == "bicubic"
    assert metadata.bicubic_rule == RULE
    assert metadata.patch_size_hr == 16
    assert metadata.num_patches == 10
    assert metadata.pool_seed == 3
    assert metadata.source_image_count == 3
    assert metadata.patch_pool_id.startswith("div2k_x4_p16_n10_s3_")

    stored = json.loads((pool_dir / "metadata.json").read_text(encoding="utf-8"))
    assert stored == {**metadata.__dict__}


def _expected_lr(hr: np.ndarray) -> np.ndarray:
    """전체 HR 영상을 공용 bicubic 규칙으로 축소하고 8비트로 반올림한 기대 LR (H, W, 3)."""
    tensor = torch.from_numpy(hr.astype(np.float64) / 255.0).permute(2, 0, 1).unsqueeze(0)
    lr = bicubic_resize(tensor, 1.0 / SCALE, antialiasing=True)
    return np.rint(lr.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def test_patch_pool_pairs_each_hr_patch_with_the_aligned_lr_region(tmp_path: Path) -> None:
    """각 패치의 HR은 원천 HR 영상의 배율 배수 좌표 영역이고, LR은 같은 영상 전체를 열화한 결과의 대응 영역이다."""
    hr_dir = tmp_path / "hr"
    paths = _write_noise_images(hr_dir, [(40, 52), (36, 44), (48, 40)])
    sources = [np.array(Image.open(p).convert("RGB")) for p in paths]
    expected_lrs = [_expected_lr(source) for source in sources]

    _make_pool(hr_dir, tmp_path / "out", num_patches=12)

    pool_dir = tmp_path / "out" / "patch_pool"
    hr_pool = np.load(pool_dir / "hr.npy")
    lr_pool = np.load(pool_dir / "lr.npy")
    patch, lr_patch = 16, 16 // SCALE
    for hr_patch, lr_region in zip(hr_pool, lr_pool):
        matches = []
        for source, expected_lr in zip(sources, expected_lrs):
            height, width = source.shape[:2]
            for top in range(0, height - patch + 1, SCALE):
                for left in range(0, width - patch + 1, SCALE):
                    if np.array_equal(source[top : top + patch, left : left + patch], hr_patch):
                        matches.append((expected_lr, top // SCALE, left // SCALE))
        assert len(matches) == 1, "잡음 영상이라 HR 패치는 배율 배수 좌표의 한 위치에서만 나와야 한다"
        expected_lr, lr_top, lr_left = matches[0]
        np.testing.assert_array_equal(
            lr_region, expected_lr[lr_top : lr_top + lr_patch, lr_left : lr_left + lr_patch]
        )


@pytest.mark.parametrize(
    ("sizes", "overrides", "message"),
    [
        ([(40, 52)], {"patch_size_hr": 18}, "배율"),
        ([(40, 52), (12, 52)], {}, "패치"),
        ([], {}, "영상"),
        ([(40, 52)], {"bicubic_rule": "pil_default"}, "bicubic"),
        ([(40, 52)], {"num_patches": 0}, "num_patches"),
    ],
    ids=["patch-not-multiple-of-scale", "image-smaller-than-patch", "no-images", "unknown-rule", "zero-patches"],
)
def test_patch_pool_rejects_invalid_inputs_before_writing(
    tmp_path: Path, sizes: list[tuple[int, int]], overrides: dict, message: str
) -> None:
    """잘못된 조건은 풀을 쓰기 전에 ValueError로 실패해, 반쯤 만든 풀이 남지 않는다."""
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    _write_noise_images(hr_dir, sizes)

    with pytest.raises(ValueError, match=message):
        _make_pool(hr_dir, tmp_path / "out", **overrides)

    assert not (tmp_path / "out" / "patch_pool").exists()


def _prepare_testsets(hr_root: Path, output_dir: Path, **overrides) -> None:
    arguments = dict(
        scale=SCALE,
        dataset_names=["SetA", "SetB"],
        degradation="bicubic",
        bicubic_rule=RULE,
    )
    arguments.update(overrides)
    prepare_testsets(hr_root, output_dir, **arguments)


def test_testsets_crop_hr_to_scale_multiple_and_store_degraded_lr(tmp_path: Path) -> None:
    """크기가 배율의 배수가 아닌 HR은 잘라(modcrop) 저장하고, LR은 잘린 HR을 열화한 결과다."""
    hr_root = tmp_path / "hr_root"
    sources = {}
    for name, sizes, seed in (("SetA", [(42, 51), (40, 52)], 1), ("SetB", [(37, 38)], 2)):
        for path in _write_noise_images(hr_root / name, sizes, seed=seed):
            sources[(name, path.stem)] = np.array(Image.open(path).convert("RGB"))

    _prepare_testsets(hr_root, tmp_path / "out")

    assert len(sources) == 3
    for (name, image_id), source in sources.items():
        height, width = source.shape[:2]
        cropped = source[: height - height % SCALE, : width - width % SCALE]
        stored_hr = np.array(Image.open(tmp_path / "out" / "testsets" / name / "HR" / f"{image_id}.png"))
        stored_lr = np.array(Image.open(tmp_path / "out" / "testsets" / name / "LR" / f"{image_id}.png"))
        np.testing.assert_array_equal(stored_hr, cropped)
        assert stored_lr.shape == (cropped.shape[0] // SCALE, cropped.shape[1] // SCALE, 3)
        np.testing.assert_array_equal(stored_lr, _expected_lr(cropped))

    stored = json.loads((tmp_path / "out" / "testsets" / "metadata.json").read_text(encoding="utf-8"))
    assert stored["scale"] == SCALE
    assert stored["degradation"] == "bicubic"
    assert stored["bicubic_rule"] == RULE
    assert stored["image_counts"] == {"SetA": 2, "SetB": 1}


@pytest.mark.parametrize(
    ("missing", "empty", "overrides", "message"),
    [
        (True, False, {}, "SetB"),
        (False, True, {}, "SetB"),
        (False, False, {"bicubic_rule": "pil_default"}, "bicubic"),
    ],
    ids=["missing-dataset-dir", "empty-dataset-dir", "unknown-rule"],
)
def test_testsets_reject_invalid_inputs_before_writing(
    tmp_path: Path, missing: bool, empty: bool, overrides: dict, message: str
) -> None:
    """데이터셋 폴더가 없거나 비었거나 규칙이 틀리면 아무것도 쓰기 전에 ValueError로 실패한다."""
    hr_root = tmp_path / "hr_root"
    _write_noise_images(hr_root / "SetA", [(40, 52)])
    if empty:
        (hr_root / "SetB").mkdir()
    elif not missing:
        _write_noise_images(hr_root / "SetB", [(40, 52)])

    with pytest.raises(ValueError, match=message):
        _prepare_testsets(hr_root, tmp_path / "out", **overrides)

    assert not (tmp_path / "out" / "testsets").exists()


def test_patch_pool_is_created_only_once_per_output_dir(tmp_path: Path) -> None:
    """이미 패치 풀이 있는 출력 폴더에는 다시 만들지 않는다(풀은 한 번만 생성해 모두가 공유한다)."""
    hr_dir = tmp_path / "hr"
    _write_noise_images(hr_dir, [(40, 52), (36, 44)])
    first = _make_pool(hr_dir, tmp_path / "out")
    stored_before = (tmp_path / "out" / "patch_pool" / "hr.npy").read_bytes()

    with pytest.raises(FileExistsError, match="patch_pool"):
        _make_pool(hr_dir, tmp_path / "out", pool_seed=99)

    assert (tmp_path / "out" / "patch_pool" / "hr.npy").read_bytes() == stored_before
    stored = json.loads((tmp_path / "out" / "patch_pool" / "metadata.json").read_text(encoding="utf-8"))
    assert stored["patch_pool_id"] == first.patch_pool_id


def test_testsets_are_created_only_once_per_output_dir(tmp_path: Path) -> None:
    """이미 테스트셋이 있는 출력 폴더에는 다시 만들지 않아, 이전 실행의 파일이 섞이지 않는다."""
    hr_root = tmp_path / "hr_root"
    for name in ("SetA", "SetB"):
        _write_noise_images(hr_root / name, [(40, 52)])
    _prepare_testsets(hr_root, tmp_path / "out")

    with pytest.raises(FileExistsError, match="testsets"):
        _prepare_testsets(hr_root, tmp_path / "out")


def test_testsets_reject_images_that_share_a_stem(tmp_path: Path) -> None:
    """img.png와 img.jpg처럼 이름(확장자 제외)이 같은 영상은 같은 출력 파일을 덮어쓰므로 거부한다."""
    hr_root = tmp_path / "hr_root"
    _write_noise_images(hr_root / "SetA", [(40, 52)])
    _write_noise_images(hr_root / "SetB", [(40, 52)])
    Image.fromarray(np.zeros((40, 52, 3), dtype=np.uint8)).save(hr_root / "SetB" / "img_000.bmp")

    with pytest.raises(ValueError, match="중복"):
        _prepare_testsets(hr_root, tmp_path / "out")

    assert not (tmp_path / "out" / "testsets").exists()


def _prepare_everything(tmp_path: Path) -> Path:
    """합성 데이터로 패치 풀과 테스트셋을 만들어 데이터 폴더(out)를 반환한다."""
    _write_noise_images(tmp_path / "div2k", [(40, 52), (36, 44)])
    for name in ("SetA", "SetB"):
        _write_noise_images(tmp_path / "hr_root" / name, [(42, 51)], seed=5)
    _make_pool(tmp_path / "div2k", tmp_path / "out")
    _prepare_testsets(tmp_path / "hr_root", tmp_path / "out")
    return tmp_path / "out"


def test_pack_archive_bundles_pool_and_testsets_into_one_zip(tmp_path: Path) -> None:
    """패치 풀과 테스트셋이 zip 1개에 담기고, 풀어 보면 압축 전 파일과 바이트 단위로 같다."""
    data_dir = _prepare_everything(tmp_path)
    archive_path = tmp_path / "archives" / "data.zip"

    returned = pack_archive(data_dir, archive_path)

    assert returned == archive_path
    assert [p.name for p in archive_path.parent.iterdir()] == ["data.zip"]
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(archive.namelist())
        archive.extractall(tmp_path / "unpacked")
    expected_names = sorted(
        p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file()
    )
    assert names == expected_names
    assert "patch_pool/metadata.json" in names and "testsets/metadata.json" in names
    for name in expected_names:
        assert (tmp_path / "unpacked" / name).read_bytes() == (data_dir / name).read_bytes()


def test_pack_archive_rejects_archive_path_inside_data_dir(tmp_path: Path) -> None:
    """압축 파일을 데이터 폴더 안에 만들면 자기 자신을 담게 되므로 거부한다."""
    data_dir = _prepare_everything(tmp_path)

    with pytest.raises(ValueError, match="데이터 폴더"):
        pack_archive(data_dir, data_dir / "data.zip")

    assert not (data_dir / "data.zip").exists()


def test_patch_pool_is_reproducible_and_depends_on_pool_seed(tmp_path: Path) -> None:
    """같은 입력·생성 시드는 같은 풀과 식별자를, 다른 생성 시드는 다른 풀과 식별자를 만든다."""
    hr_dir = tmp_path / "hr"
    _write_noise_images(hr_dir, [(40, 52), (36, 44), (48, 40)])

    first = _make_pool(hr_dir, tmp_path / "a")
    second = _make_pool(hr_dir, tmp_path / "b")
    other_seed = _make_pool(hr_dir, tmp_path / "c", pool_seed=4)

    assert first == second
    for name in ("hr.npy", "lr.npy"):
        np.testing.assert_array_equal(
            np.load(tmp_path / "a" / "patch_pool" / name), np.load(tmp_path / "b" / "patch_pool" / name)
        )
    assert other_seed.patch_pool_id != first.patch_pool_id
    assert other_seed.content_sha256 != first.content_sha256
