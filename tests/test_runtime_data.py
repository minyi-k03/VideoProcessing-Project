"""실행 창구의 데이터 준비(T3.8) 바깥 동작 테스트 (SPEC.md US5, 7, 38).

Drive에서 복사한 zip 해제(Colab)와 이미 해제된 입력 폴더(Kaggle)를 같은 `--data-dir` 인자로 다룬다.
작은 합성 zip만 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from sr_pipeline.engine.runtime_data import extract_data_archive, find_data_root, sha256_of


def make_archive(path: Path, *, pool_id: str = "pool_a", prefix: str = "", extra: dict[str, bytes] | None = None) -> Path:
    """`data.prepare.pack_archive`와 같은 구조(`patch_pool/...`, `testsets/...`)의 작은 zip을 만든다."""
    members = {
        "patch_pool/metadata.json": json.dumps({"patch_pool_id": pool_id, "scale": 4}).encode(),
        "patch_pool/hr.npy": b"hr-bytes",
        "testsets/Set5/HR/a.png": b"png",
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(prefix + name, content)
    return path


def test_extract_unpacks_archive_into_data_dir_and_find_data_root_locates_it(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "pool.zip")
    data_dir = tmp_path / "data"

    root = extract_data_archive(archive, data_dir)

    assert root == data_dir
    assert (data_dir / "testsets" / "Set5" / "HR" / "a.png").read_bytes() == b"png"
    assert find_data_root(data_dir) == data_dir


def test_find_data_root_finds_pool_nested_below_the_input_folder(tmp_path: Path) -> None:
    """Kaggle은 Dataset 폴더 아래에 한두 단계를 더 감싸서 붙인다."""
    nested = tmp_path / "input" / "datasets" / "someone" / "slug"
    make_archive(tmp_path / "pool.zip")
    extract_data_archive(tmp_path / "pool.zip", nested)

    assert find_data_root(tmp_path / "input") == nested


def test_find_data_root_rejects_missing_and_ambiguous_pools(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="patch_pool"):
        find_data_root(tmp_path / "empty")
    for name in ("one", "two"):
        extract_data_archive(make_archive(tmp_path / f"{name}.zip"), tmp_path / "both" / name)
    with pytest.raises(ValueError, match="둘 이상") as error:
        find_data_root(tmp_path / "both")
    assert str(tmp_path) not in str(error.value)  # Kaggle Dataset 경로가 노트북 출력에 남지 않게 경로는 싣지 않는다


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256_of(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


@pytest.mark.parametrize("member", ["../escape.txt", "/abs/escape.txt", "patch_pool/../../escape.txt"])
def test_extract_rejects_members_that_escape_the_data_dir(tmp_path: Path, member: str) -> None:
    archive = make_archive(tmp_path / "evil.zip", extra={member: b"x"})
    data_dir = tmp_path / "work" / "data"

    with pytest.raises(ValueError, match="밖"):
        extract_data_archive(archive, data_dir)

    assert not data_dir.exists()
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "work" / "escape.txt").exists()


def test_extract_verifies_archive_sha256_before_unpacking(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "pool.zip")
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="sha256"):
        extract_data_archive(archive, data_dir, expected_sha256="0" * 64)
    assert not data_dir.exists()

    extract_data_archive(archive, data_dir, expected_sha256=sha256_of(archive).upper())  # 대소문자는 무시
    assert find_data_root(data_dir) == data_dir


def test_extract_rejects_corrupt_archive_and_leaves_nothing_behind(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    data_dir = tmp_path / "data"

    with pytest.raises(zipfile.BadZipFile):
        extract_data_archive(archive, data_dir)
    assert not data_dir.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_failure_while_unpacking_leaves_no_partial_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = make_archive(tmp_path / "pool.zip")
    data_dir = tmp_path / "data"

    def broken_extract(self, member, path=None, pwd=None):
        raise OSError("디스크가 가득 찼습니다")

    monkeypatch.setattr(zipfile.ZipFile, "extract", broken_extract)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda self, *a, **k: broken_extract(self, None))
    with pytest.raises(OSError):
        extract_data_archive(archive, data_dir)

    assert not data_dir.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_extract_reuses_already_unpacked_data_of_the_same_pool(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "pool.zip")
    data_dir = tmp_path / "data"
    extract_data_archive(archive, data_dir)
    marker = data_dir / "local_marker.txt"
    marker.write_text("keep", encoding="utf-8")

    assert extract_data_archive(archive, data_dir) == data_dir
    assert marker.read_text(encoding="utf-8") == "keep"  # 다시 풀지 않는다


def test_extract_refuses_to_mix_a_different_pool_into_existing_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    extract_data_archive(make_archive(tmp_path / "a.zip", pool_id="pool_a"), data_dir)

    with pytest.raises(ValueError, match="pool_b"):
        extract_data_archive(make_archive(tmp_path / "b.zip", pool_id="pool_b"), data_dir)
    assert json.loads((data_dir / "patch_pool" / "metadata.json").read_text(encoding="utf-8"))["patch_pool_id"] == "pool_a"


def test_extract_rejects_archive_without_pool_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "nopool.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("testsets/Set5/HR/a.png", b"png")

    with pytest.raises(ValueError, match="patch_pool/metadata.json"):
        extract_data_archive(archive, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_extract_refuses_reuse_when_same_pool_id_archive_differs_in_testsets(tmp_path: Path) -> None:
    """패치 풀 ID가 같더라도 테스트셋 등 압축 내용이 다른 ZIP이면 기존 해제본을 재사용하지 않는다."""
    data_dir = tmp_path / "data"
    first = make_archive(tmp_path / "first.zip", pool_id="pool_a", extra={"testsets/Set5/HR/a.png": b"old"})
    second = make_archive(tmp_path / "second.zip", pool_id="pool_a", extra={"testsets/Set5/HR/a.png": b"new"})
    extract_data_archive(first, data_dir)

    with pytest.raises(ValueError, match="테스트셋"):
        extract_data_archive(second, data_dir)
    assert (data_dir / "testsets" / "Set5" / "HR" / "a.png").read_bytes() == b"old"


def test_extract_refuses_reuse_when_archive_marker_is_missing(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "pool.zip", pool_id="pool_a")
    data_dir = tmp_path / "data"
    extract_data_archive(archive, data_dir)
    (data_dir / ".archive_sha256").unlink()

    with pytest.raises(ValueError, match="기존 해제본"):
        extract_data_archive(archive, data_dir)
