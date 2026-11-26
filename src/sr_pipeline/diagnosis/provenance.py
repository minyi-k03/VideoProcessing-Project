"""진단 실행 기록의 코드 커밋 확인 (Task T2.5 후속).

`diagnosis_settings.json`의 `commit`(SPEC 실행 기록 필드 이름)이 실제로 결과를 만든 코드를 가리키도록,
코드·설정에 커밋되지 않은 변경이 있으면 멈춘다. T3.7에서 `engine.trainer.check_git_clean`이 구현되면
본 실행과 같은 규칙을 쓰도록 그쪽으로 합친다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# 결과에 영향을 주는 코드·설정 경로. results/는 진단·학습 결과가 생기는 곳이라 확인하지 않는다.
CODE_PATHS = ("src", "scripts", "configs")
SR_GENERATION_SETTINGS_FILENAME = "sr_generation_settings.json"


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("git을 찾을 수 없습니다. git이 설치된 환경의 저장소에서 실행하세요.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"git {' '.join(args)} 실패({repo_root}): {error.stderr.strip()}") from error
    return result.stdout.strip()


def resolve_code_commit(repo_root: Path, *, allow_dirty: bool = False) -> str:
    """`repo_root` 저장소의 HEAD 커밋을 돌려줍니다.

    `src/`·`scripts/`·`configs/`에 커밋되지 않은 변경(추적되지 않는 새 파일 포함)이 있으면 `RuntimeError`.
    `allow_dirty=True`면 멈추지 않고 `"<커밋>-dirty"`를 돌려줘, 기록만 보고도 깨끗하지 않은 실행임을 알 수 있다.
    """
    commit = _git(repo_root, "rev-parse", "HEAD")
    changes = _git(repo_root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    if not changes:
        return commit
    if allow_dirty:
        return f"{commit}-dirty"
    raise RuntimeError(
        "코드·설정에 커밋되지 않은 변경이 있어 결과를 커밋으로 추적할 수 없습니다. "
        f"커밋한 뒤 다시 실행하세요(그래도 실행하려면 allow_dirty):\n{changes}"
    )
