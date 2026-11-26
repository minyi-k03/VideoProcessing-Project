"""실행 창구의 런타임 데이터 준비 (T3.8, SPEC.md US5·7·38, ADR-0005).

- Colab: 공유 Drive의 압축 파일 1개를 런타임 로컬로 복사한 뒤 `extract_data_archive`로 푼다.
- Kaggle: Dataset이 `/kaggle/input/...`에 (자동 해제돼) 붙으므로 `find_data_root`로 그 폴더를 그대로 쓴다.
데이터를 만드는 규칙(`data/prepare.py`)은 바꾸지 않는다. 여기서는 이미 만들어진 압축 파일을 풀기만 한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from sr_pipeline.data.prepare import METADATA_FILENAME, POOL_DIRNAME

_POOL_METADATA_MEMBER = f"{POOL_DIRNAME}/{METADATA_FILENAME}"
MAX_SEARCH_DEPTH = 4  # Kaggle은 /kaggle/input/datasets/<유저>/<slug>처럼 몇 단계를 더 감싼다.


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_data_root(path: Path) -> Path:
    """`patch_pool/metadata.json`을 가진 폴더(데이터 루트)를 `path` 자신 또는 그 아래에서 찾는다.

    없으면 `FileNotFoundError`, 둘 이상이면(어느 풀인지 알 수 없다) `ValueError`.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {path}")
    base_depth = len(path.parts)
    roots: list[Path] = []
    for current, dirnames, _ in os.walk(path):
        current_path = Path(current)
        if (current_path / POOL_DIRNAME / METADATA_FILENAME).is_file():
            roots.append(current_path)
            dirnames[:] = []  # 데이터 루트 안쪽은 더 찾지 않는다.
        elif len(current_path.parts) - base_depth >= MAX_SEARCH_DEPTH:
            dirnames[:] = []
    if not roots:
        raise FileNotFoundError(f"{path} 아래에서 {POOL_DIRNAME}/{METADATA_FILENAME}을 찾지 못했습니다.")
    if len(roots) > 1:
        raise ValueError(f"{path} 아래에 패치 풀이 둘 이상 있습니다(하나만 두세요): {sorted(str(r) for r in roots)}")
    return roots[0]


def _check_member_names(archive: zipfile.ZipFile) -> None:
    """압축 안의 경로가 해제 폴더 밖을 가리키면(절대 경로, `..`, 드라이브 문자) 거부한다(zip-slip)."""
    for name in archive.namelist():
        normalized = name.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if normalized.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            raise ValueError(f"압축 파일의 경로가 해제 폴더 밖을 가리킵니다: {name!r}")


def _archive_pool_id(archive: zipfile.ZipFile) -> str:
    if _POOL_METADATA_MEMBER not in archive.namelist():
        raise ValueError(f"압축 파일에 {_POOL_METADATA_MEMBER}가 없습니다: 패치 풀과 테스트셋을 묶은 압축 파일이 아닙니다.")
    metadata = json.loads(archive.read(_POOL_METADATA_MEMBER).decode("utf-8"))
    return metadata["patch_pool_id"]


def extract_data_archive(archive: Path, data_dir: Path, *, expected_sha256: str | None = None) -> Path:
    """압축 파일을 `data_dir`에 풀고 데이터 루트(`data_dir`)를 돌려준다.

    - `expected_sha256`이 있으면 풀기 전에 압축 파일 해시를 확인한다(Drive 대체 경로의 `gdown` 내려받기 검증).
    - `data_dir`에 같은 패치 풀이 이미 풀려 있으면 다시 풀지 않고 재사용한다(세션 재시작). 다른 풀이면 섞이지 않게 거부한다.
    - 옆의 `<data_dir>.partial`에 다 푼 뒤 이름을 바꾸므로 실패해도 반쪽 `data_dir`이 남지 않는다.
    """
    archive, data_dir = Path(archive), Path(data_dir)
    if expected_sha256 is not None and sha256_of(archive) != expected_sha256.strip().lower():
        raise ValueError("압축 파일의 sha256이 기대값과 다릅니다. 내려받은 파일이 원본과 다르니 다시 받으세요.")
    with zipfile.ZipFile(archive) as zf:
        _check_member_names(zf)
        pool_id = _archive_pool_id(zf)
        existing = data_dir / POOL_DIRNAME / METADATA_FILENAME
        if existing.is_file():
            existing_id = json.loads(existing.read_text(encoding="utf-8")).get("patch_pool_id")
            if existing_id != pool_id:
                raise ValueError(
                    f"{data_dir}에 이미 다른 패치 풀({existing_id})이 있어 {pool_id}를 풀 수 없습니다. 다른 --data-dir을 쓰세요."
                )
            return data_dir
        if data_dir.exists() and any(data_dir.iterdir()):
            raise ValueError(f"{data_dir}이 비어 있지 않고 패치 풀도 없습니다. 비어 있는 --data-dir을 쓰세요.")
        if zf.testzip() is not None:
            raise zipfile.BadZipFile("압축 파일이 손상되었습니다.")
        staging = data_dir.with_name(data_dir.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)  # 이전에 끊긴 해제가 남긴 임시 폴더
        try:
            staging.mkdir(parents=True)
            for member in zf.namelist():
                zf.extract(member, staging)
            if data_dir.exists():
                data_dir.rmdir()  # 비어 있는 폴더만 여기까지 온다
            staging.replace(data_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return data_dir
