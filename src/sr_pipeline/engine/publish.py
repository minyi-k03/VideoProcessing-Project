"""실행 기록 커밋·push (T3.8, SPEC.md US3·40, ADR-0005).

노트북은 이 모듈을 부르기만 한다. 토큰은 인자로만 받아 임시 `GIT_ASKPASS` 스크립트가 환경 변수로 읽게 한다.
remote URL·`.git/config`·명령행·출력·예외 메시지 어디에도 남기지 않는다(저장소가 Public이다).
스크립트는 POSIX `sh`용이다(Colab·Kaggle). 로컬 Windows에서는 push 대신 `bundle_run_record`나 자신의 git 자격 증명을 쓴다.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

RUNS_RELATIVE = Path("results") / "runs"
REQUIRED_RECORD_FILES = ("summary.csv", "per_image.csv")
TRAIN_LOG_FILENAME = "train_log.json"  # 학습하는 구성에만 있다(Bicubic 하한에는 없다).
WEIGHTS_FILENAME = "weights.pt"
_MASK = "***"
_ASKPASS_SCRIPT = (
    '#!/bin/sh\ncase "$1" in\n  Username*) printf "%s\n" "$SR_PUSH_USER";;\n  *) printf "%s\n" "$SR_PUSH_TOKEN";;\nesac\n'
)


class PushFailed(RuntimeError):
    """push가 실패했다. 커밋은 로컬에 남아 있다. 메시지에서 토큰은 가려져 있다."""


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} 실패: {result.stderr.strip()}")
    return result.stdout.strip()


def _check_run_dir(repo_root: Path, run_dir: Path, trained: bool) -> Path:
    runs_dir = (repo_root / RUNS_RELATIVE).resolve()
    resolved = Path(run_dir).resolve()
    if resolved.parent != runs_dir:
        raise ValueError(f"실행 기록 폴더는 {RUNS_RELATIVE.as_posix()}/<run_id> 여야 합니다: {run_dir}")
    for name in (*REQUIRED_RECORD_FILES, *((TRAIN_LOG_FILENAME,) if trained else ())):
        if not (resolved / name).is_file():
            raise ValueError(f"{run_dir}에 {name}이 없어 완성된 실행 기록이 아닙니다.")
    return resolved


def commit_run_record(
    repo_root: Path,
    run_dir: Path,
    *,
    message: str,
    author_name: str,
    author_email: str,
    weights_path: Path | None = None,
    trained: bool = True,
) -> str:
    """실행 기록 폴더 하나만 커밋하고 커밋 해시를 돌려준다. 다른 변경은 건드리지 않는다.

    `run_dir`은 `<repo_root>/results/runs/<run_id>`여야 한다(실행 CLI의 기본 `--output-dir`). `trained`는 학습하는 구성이면
    True(기본)이고, 그때는 `train_log.json`도 있어야 한다. 학습 없는 구성(Bicubic 하한)만 `trained=False`로 부른다.

    `weights_path`는 Drive를 쓸 수 없는 ADR-0005 예외에서만 준다. 평가 체크포인트를 기록 폴더로 복사해 함께 커밋한다
    (`.gitignore`의 `*.pt`를 넘어 강제로 추가). 커밋자 신원은 이번 커밋에만 쓰고 저장소 설정에 남기지 않는다.
    """
    repo_root = Path(repo_root)
    resolved = _check_run_dir(repo_root, run_dir, trained)
    if weights_path is not None:
        shutil.copyfile(weights_path, resolved / WEIGHTS_FILENAME)
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    _git(repo_root, "add", "--force" if weights_path is not None else "--", relative)
    _git(
        repo_root, "-c", f"user.name={author_name}", "-c", f"user.email={author_email}",
        "commit", "-q", "-m", message, "--", relative,
    )
    return _git(repo_root, "rev-parse", "HEAD")


def push_branch(repo_root: Path, branch: str, *, username: str, token: str, remote: str = "origin") -> None:
    """`branch`를 `remote`에 push한다. 실패하면 토큰을 가린 메시지로 `PushFailed`."""
    if not username or not token:
        raise ValueError("사용자명과 토큰이 모두 필요합니다.")
    handle, askpass_path = tempfile.mkstemp(suffix=".sh", prefix="sr_askpass_")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(_ASKPASS_SCRIPT)
        os.chmod(askpass_path, stat.S_IRWXU)
        env = {
            **os.environ,
            "GIT_ASKPASS": askpass_path,
            "GIT_TERMINAL_PROMPT": "0",
            "SR_PUSH_USER": username,
            "SR_PUSH_TOKEN": token,
        }
        result = subprocess.run(
            ["git", "-C", str(repo_root), "-c", "credential.helper=", "push", remote, branch],
            env=env, capture_output=True, text=True,
        )
    finally:
        Path(askpass_path).unlink(missing_ok=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).replace(token, _MASK).strip()
        raise PushFailed(
            f"{remote}에 {branch}를 push하지 못했습니다(커밋은 로컬에 남아 있습니다). "
            f"실행 기록을 내려받아 로컬에서 커밋하세요(ADR-0005 대체 경로): {detail}"
        )


def bundle_run_record(run_dir: Path, dest_zip: Path) -> Path:
    """push가 안 될 때 내려받아 로컬에서 커밋하도록 실행 기록 폴더를 zip 하나로 묶는다(`<run_id>/...` 상대 경로)."""
    run_dir, dest_zip = Path(run_dir), Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(run_dir.name) / path.relative_to(run_dir)).as_posix())
    return dest_zip
