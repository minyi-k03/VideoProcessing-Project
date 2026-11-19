"""진단 실행 커밋 확인 바깥 동작 테스트 (Task T2.5 후속).

`tmp_path`에 작은 git 저장소를 만들어 검증한다. git이 없는 환경에서는 건너뛴다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sr_pipeline.diagnosis.provenance import resolve_code_commit

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없는 환경")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in ("src/module.py", "scripts/run.py", "configs/rules.yaml", "results/pilot/.gitkeep"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x = 1\n", encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "add", ".")
    _git(root, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "init")
    return root


def test_clean_repo_returns_head(repo: Path) -> None:
    assert resolve_code_commit(repo) == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("relative", ["src/module.py", "scripts/run.py", "configs/rules.yaml", "src/new_module.py"])
def test_uncommitted_code_or_config_change_is_rejected(repo: Path, relative: str) -> None:
    (repo / relative).write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        resolve_code_commit(repo)


def test_result_files_do_not_count_as_code_changes(repo: Path) -> None:
    """진단·학습 결과가 생기는 results/는 코드가 아니므로 확인 대상이 아니다."""
    (repo / "results" / "pilot" / "pilot_train_settings.json").write_text("{}", encoding="utf-8")
    (repo / "results" / "pilot" / ".gitkeep").write_text("changed", encoding="utf-8")
    assert resolve_code_commit(repo) == _git(repo, "rev-parse", "HEAD")


def test_allow_dirty_marks_the_commit(repo: Path) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert resolve_code_commit(repo, allow_dirty=True) == _git(repo, "rev-parse", "HEAD") + "-dirty"


def test_not_a_git_repository_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        resolve_code_commit(tmp_path)
