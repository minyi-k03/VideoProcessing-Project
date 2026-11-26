"""실행 기록 커밋·push(T3.8) 바깥 동작 테스트 (SPEC.md US3, 40, ADR-0005).

임시 git 저장소와 로컬 bare 저장소만 쓴다. 토큰은 가짜 문자열이며 어디에도 남지 않아야 한다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from sr_pipeline.engine.publish import PushFailed, bundle_run_record, commit_run_record, push_branch

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없는 환경")

RUN_ID = "0123abc-baseline-s0"
FAKE_TOKEN = "ghp_FAKE_TOKEN_for_tests_1234567890"
IDENTITY = {"author_name": "Test Author", "author_email": "test@example.com"}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "work")
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return root


def write_run_record(repo: Path, run_id: str = RUN_ID, *, train_log: bool = True) -> Path:
    run_dir = repo / "results" / "runs" / run_id
    (run_dir / "images" / "Set5").mkdir(parents=True)
    (run_dir / "summary.csv").write_text("run_id,psnr_y\n" + run_id + ",30.0\n", encoding="utf-8")
    (run_dir / "per_image.csv").write_text("run_id,image_id\n" + run_id + ",a\n", encoding="utf-8")
    (run_dir / "images" / "Set5" / "a_median.png").write_bytes(b"png")
    if train_log:
        (run_dir / "train_log.json").write_text("{}\n", encoding="utf-8")
    return run_dir


def committed_files(repo: Path) -> set[str]:
    return set(git(repo, "show", "--name-only", "--pretty=format:", "HEAD").splitlines())


def test_commit_adds_only_the_run_record_folder(repo: Path) -> None:
    run_dir = write_run_record(repo)
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")  # 실행과 무관한 변경은 커밋하지 않는다
    (repo / "notes.txt").write_text("untracked", encoding="utf-8")

    sha = commit_run_record(repo, run_dir, message="results: baseline seed 0", **IDENTITY)

    assert sha == git(repo, "rev-parse", "HEAD")
    assert committed_files(repo) == {
        f"results/runs/{RUN_ID}/{name}"
        for name in ("summary.csv", "per_image.csv", "train_log.json", "images/Set5/a_median.png")
    }
    assert git(repo, "log", "-1", "--format=%an <%ae>|%s") == "Test Author <test@example.com>|results: baseline seed 0"
    status = git(repo, "status", "--porcelain").splitlines()
    assert {line.lstrip() for line in status} == {"M src/module.py", "?? notes.txt"}  # 무관한 변경은 그대로 남아 있다


def test_commit_does_not_change_the_repositorys_git_identity(repo: Path) -> None:
    run_dir = write_run_record(repo)
    commit_run_record(repo, run_dir, message="m", **IDENTITY)

    result = subprocess.run(["git", "-C", str(repo), "config", "--local", "--get", "user.email"], capture_output=True, text=True)
    assert result.stdout.strip() == ""  # 신원은 이번 커밋에만 쓰고 저장소 설정에 남기지 않는다


@pytest.mark.parametrize("case", ["outside_runs", "missing_summary", "missing_per_image", "missing_run_dir"])
def test_commit_rejects_folders_that_are_not_complete_run_records(repo: Path, case: str) -> None:
    run_dir = write_run_record(repo)
    if case == "outside_runs":
        run_dir = repo / "results" / "pilot"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.csv").write_text("x\n", encoding="utf-8")
        (run_dir / "per_image.csv").write_text("x\n", encoding="utf-8")
    elif case == "missing_summary":
        (run_dir / "summary.csv").unlink()
    elif case == "missing_per_image":
        (run_dir / "per_image.csv").unlink()
    else:
        run_dir = repo / "results" / "runs" / "nope"
    head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError):
        commit_run_record(repo, run_dir, message="m", **IDENTITY)
    assert git(repo, "rev-parse", "HEAD") == head


def test_bicubic_record_without_train_log_is_accepted_only_when_declared_untrained(repo: Path) -> None:
    """학습 없는 구성(Bicubic 하한)은 `train_log.json`이 없다. 학습하는 구성의 기록에 이게 없으면 반쪽 기록이다."""
    run_dir = write_run_record(repo, "0123abc-bicubic-s0", train_log=False)
    head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="train_log.json"):
        commit_run_record(repo, run_dir, message="m", **IDENTITY)
    assert git(repo, "rev-parse", "HEAD") == head

    commit_run_record(repo, run_dir, message="m", trained=False, **IDENTITY)
    assert "results/runs/0123abc-bicubic-s0/summary.csv" in committed_files(repo)


def test_weights_are_committed_only_when_requested(repo: Path, tmp_path: Path) -> None:
    weights = tmp_path / "ckpt" / RUN_ID / "weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    (repo / ".gitignore").write_text("*.pt\n", encoding="utf-8")  # 실제 저장소도 *.pt를 무시한다

    plain = commit_run_record(repo, write_run_record(repo), message="no weights", **IDENTITY)
    assert plain and not any(name.endswith("weights.pt") for name in committed_files(repo))

    run_dir = write_run_record(repo, "0123abc-B-s0")
    commit_run_record(repo, run_dir, message="with weights", weights_path=weights, **IDENTITY)
    assert "results/runs/0123abc-B-s0/weights.pt" in committed_files(repo)
    assert (run_dir / "weights.pt").read_bytes() == b"weights"


def test_committing_weights_does_not_force_in_other_ignored_files(repo: Path, tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    (repo / ".gitignore").write_text("*.pt\n*.tmp\n", encoding="utf-8")
    run_dir = write_run_record(repo)
    (run_dir / "scratch.tmp").write_text("ignored", encoding="utf-8")

    commit_run_record(repo, run_dir, message="with weights", weights_path=weights, **IDENTITY)

    files = committed_files(repo)
    assert f"results/runs/{RUN_ID}/weights.pt" in files
    assert f"results/runs/{RUN_ID}/scratch.tmp" not in files


def test_failed_commit_leaves_no_copied_weights_and_nothing_staged(repo: Path, tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")  # 커밋을 실패시킨다
    hook.chmod(0o755)
    run_dir = write_run_record(repo)
    head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(RuntimeError):
        commit_run_record(repo, run_dir, message="m", weights_path=weights, **IDENTITY)

    assert git(repo, "rev-parse", "HEAD") == head
    assert not (run_dir / "weights.pt").exists()
    assert git(repo, "diff", "--cached", "--name-only") == ""


@pytest.fixture
def remote(tmp_path: Path, repo: Path) -> Path:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "origin", str(bare))
    return bare


def test_push_branch_publishes_the_commit_and_leaves_no_token_behind(
    repo: Path, remote: Path, tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    commit_run_record(repo, write_run_record(repo), message="results", **IDENTITY)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))  # 임시 askpass 스크립트가 만들어지는 곳을 이 테스트 전용으로

    push_branch(repo, "work", username="someone", token=FAKE_TOKEN)

    assert git(remote, "rev-parse", "work") == git(repo, "rev-parse", "HEAD")
    assert FAKE_TOKEN not in (repo / ".git" / "config").read_text(encoding="utf-8")
    assert FAKE_TOKEN not in git(repo, "remote", "-v")
    captured = capfd.readouterr()
    assert FAKE_TOKEN not in captured.out + captured.err
    assert list(temp_dir.iterdir()) == []  # askpass 임시 스크립트를 지웠다


def test_push_failure_raises_push_failed_without_token_and_keeps_the_commit(repo: Path, tmp_path: Path) -> None:
    git(repo, "remote", "add", "origin", str(tmp_path / "does-not-exist.git"))
    sha = commit_run_record(repo, write_run_record(repo), message="results", **IDENTITY)

    with pytest.raises(PushFailed) as error:
        push_branch(repo, "work", username="someone", token=FAKE_TOKEN)

    assert FAKE_TOKEN not in str(error.value)
    assert git(repo, "rev-parse", "HEAD") == sha  # 커밋은 로컬에 남아 내려받아 커밋하는 대체 경로가 가능하다


def test_recommitting_an_already_committed_record_reuses_the_commit_so_push_can_be_retried(
    repo: Path, remote: Path
) -> None:
    """push가 실패해 커밋 셀을 다시 실행해도 막히지 않고, 새 커밋 없이 같은 커밋을 돌려준다."""
    run_dir = write_run_record(repo)
    first = commit_run_record(repo, run_dir, message="results", **IDENTITY)

    second = commit_run_record(repo, run_dir, message="results", **IDENTITY)

    assert second == first == git(repo, "rev-parse", "HEAD")
    push_branch(repo, "work", username="someone", token=FAKE_TOKEN)
    assert git(remote, "rev-parse", "work") == first


def test_push_failure_message_masks_a_token_echoed_by_git(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """원격이 토큰을 에러에 되돌려 줘도 메시지에서 가린다."""
    import sr_pipeline.engine.publish as publish

    def noisy_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", f"fatal: auth failed for {FAKE_TOKEN}")

    monkeypatch.setattr(publish.subprocess, "run", noisy_run)
    with pytest.raises(PushFailed) as error:
        push_branch(repo, "work", username="someone", token=FAKE_TOKEN)
    assert FAKE_TOKEN not in str(error.value) and "***" in str(error.value)


def test_push_rejects_empty_credentials(repo: Path) -> None:
    with pytest.raises(ValueError):
        push_branch(repo, "work", username="", token=FAKE_TOKEN)
    with pytest.raises(ValueError):
        push_branch(repo, "work", username="someone", token="")


def test_bundle_run_record_zips_the_folder_with_relative_paths(repo: Path, tmp_path: Path) -> None:
    run_dir = write_run_record(repo)
    dest = tmp_path / "out" / "record.zip"

    assert bundle_run_record(run_dir, dest) == dest

    with zipfile.ZipFile(dest) as archive:
        assert set(archive.namelist()) == {
            f"{RUN_ID}/summary.csv",
            f"{RUN_ID}/per_image.csv",
            f"{RUN_ID}/train_log.json",
            f"{RUN_ID}/images/Set5/a_median.png",
        }
