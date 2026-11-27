"""실행 기록 커밋·push (T3.8, SPEC.md US3·40, ADR-0005).

노트북은 이 모듈을 부르기만 한다. 토큰은 인자로만 받아 임시 `GIT_ASKPASS` 스크립트가 환경 변수로 읽게 한다.
remote URL·`.git/config`·명령행·출력·예외 메시지 어디에도 남기지 않는다(저장소가 Public이다).
스크립트는 POSIX `sh`용이다(Colab·Kaggle). 로컬 Windows에서는 push 대신 `bundle_run_record`나 자신의 git 자격 증명을 쓴다.
"""

from __future__ import annotations

import base64
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
_TRACE_ENV_PREFIXES = ("GIT_TRACE", "GIT_CURL_VERBOSE", "GCM_TRACE")
_ASKPASS_SCRIPT = r"""#!/bin/sh
case "$1" in
  Username*) printf '%s\n' "$SR_PUSH_USER";;
  *) printf '%s\n' "$SR_PUSH_TOKEN";;
esac
"""


class PushFailed(RuntimeError):
    """push가 실패했다. 커밋은 로컬에 남아 있다. 메시지에서 토큰은 가려져 있다."""


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        command = next(arg for arg in args if not arg.startswith("-") and "=" not in arg)  # `-c k=v` 다음의 하위 명령
        raise RuntimeError(f"git {command} 실패: {(result.stderr or result.stdout).strip()}")
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
    (`.gitignore`의 `*.pt`를 넘어 그 파일만 강제로 추가). 커밋자 신원은 이번 커밋에만 쓰고 저장소 설정에 남기지 않는다.

    이미 커밋된 기록을 다시 부르면 새 커밋 없이 그 기록을 마지막으로 커밋한 해시를 돌려준다(push 재시도).
    커밋이 실패하면 스테이징을 되돌리고 이번에 복사한 체크포인트를 지운다.
    """
    repo_root = Path(repo_root)
    resolved = _check_run_dir(repo_root, run_dir, trained)
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    copied_weights: Path | None = None
    if weights_path is not None:
        target = resolved / WEIGHTS_FILENAME
        if not target.exists():
            copied_weights = target
        shutil.copyfile(weights_path, target)
    try:
        _git(repo_root, "add", "--", relative)
        if weights_path is not None:  # `.gitignore`의 `*.pt`를 넘는 강제 추가는 이 파일 하나만
            _git(repo_root, "add", "--force", "--", f"{relative}/{WEIGHTS_FILENAME}")
        if _nothing_staged(repo_root, relative):
            # 이미 커밋된 기록이다(push가 실패해 커밋 셀을 다시 실행한 경우). 새 커밋 없이 그 커밋을 돌려준다.
            last = _git(repo_root, "log", "-1", "--format=%H", "--", relative)
            if not last:
                raise ValueError(f"{relative}에 커밋할 파일이 없습니다(모두 .gitignore 대상인지 확인하세요).")
            return last
        _git(
            repo_root, "-c", f"user.name={author_name}", "-c", f"user.email={author_email}",
            "commit", "-q", "-m", message, "--", relative,
        )
    except BaseException:
        # 반쪽 상태를 남기지 않는다: 스테이징을 되돌리고 이번에 복사한 체크포인트를 지운다.
        subprocess.run(["git", "-C", str(repo_root), "reset", "-q", "--", relative], capture_output=True)
        if copied_weights is not None:
            copied_weights.unlink(missing_ok=True)
        raise
    return _git(repo_root, "rev-parse", "HEAD")


def _nothing_staged(repo_root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--", relative], capture_output=True
    )
    return result.returncode == 0


def _sanitize_secret_text(text: str, *, username: str, token: str) -> str:
    """오류 메시지에서 토큰 원문과 Basic 인증(base64) 문자열을 가린다."""
    basic_auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    sanitized = text.replace(token, _MASK).replace(basic_auth, _MASK)
    return sanitized


def push_branch(repo_root: Path, branch: str, *, username: str, token: str, remote: str = "origin") -> None:
    """`branch`를 `remote`에 push한다. 실패하면 토큰을 가린 메시지로 `PushFailed`."""
    if not username or not token:
        raise ValueError("사용자명과 토큰이 모두 필요합니다.")
    handle, askpass_path = tempfile.mkstemp(suffix=".sh", prefix="sr_askpass_")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:  # Windows에서도 CRLF 없이 쓴다
            file.write(_ASKPASS_SCRIPT)
        os.chmod(askpass_path, stat.S_IRWXU)
        # 호출 환경의 Git 추적 설정(GIT_TRACE*, GIT_CURL_VERBOSE, GCM_TRACE*)이 인증 정보를 남기지 않도록 제거한다.
        base_env = {k: v for k, v in os.environ.items() if not k.startswith(_TRACE_ENV_PREFIXES)}
        env = {
            **base_env,
            "GIT_ASKPASS": askpass_path,
            "GIT_TERMINAL_PROMPT": "0",
            "SR_PUSH_USER": username,
            "SR_PUSH_TOKEN": token,
        }
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "-c", "credential.helper=", "push", remote, branch],
                env=env, capture_output=True, text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = _sanitize_secret_text(str(error), username=username, token=token)
            raise PushFailed(
                f"{remote}에 {branch}를 push하지 못했습니다(커밋은 로컬에 남아 있습니다). "
                f"실행 기록을 내려받아 로컬에서 커밋하세요(ADR-0005 대체 경로): {detail}"
            ) from None
    finally:
        Path(askpass_path).unlink(missing_ok=True)
    if result.returncode != 0:
        detail = _sanitize_secret_text((result.stderr or result.stdout).strip(), username=username, token=token)
        raise PushFailed(
            f"{remote}에 {branch}를 push하지 못했습니다(커밋은 로컬에 남아 있습니다). "
            f"실행 기록을 내려받아 로컬에서 커밋하세요(ADR-0005 대체 경로): {detail}"
        ) from None


def bundle_run_record(run_dir: Path, dest_zip: Path) -> Path:
    """push가 안 될 때 내려받아 로컬에서 커밋하도록 실행 기록 폴더를 zip 하나로 묶는다(`<run_id>/...` 상대 경로)."""
    run_dir, dest_zip = Path(run_dir), Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(run_dir.name) / path.relative_to(run_dir)).as_posix())
    return dest_zip
