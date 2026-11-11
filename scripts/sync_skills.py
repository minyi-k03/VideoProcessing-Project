#!/usr/bin/env python3
"""`.agents/skills/`(원본)를 `.claude/skills/`(사본)로 동기화한다.

사용법:
  python scripts/sync_skills.py           # 동기화 (사본에만 있는 파일이 있으면 중단)
  python scripts/sync_skills.py --check   # 불일치 시 exit 1 (CI·verify용)
  python scripts/sync_skills.py --force   # 사본에만 있는 파일까지 지우고 동기화
"""
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # scripts/의 상위 = 저장소 루트
SRC = ROOT / ".agents" / "skills"
DST = ROOT / ".claude" / "skills"


def differs(a: Path, b: Path) -> bool:
    if not b.is_dir():
        return True
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return True
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return True
    return any(differs(a / d, b / d) for d in cmp.common_dirs)


def copy_only_items(src: Path, dst: Path) -> list[Path]:
    """원본에 없고 사본에만 있는 최상위 항목(파일·폴더) 목록."""
    if not dst.is_dir():
        return []
    items = []
    for path in sorted(dst.rglob("*")):
        rel = path.relative_to(dst)
        if not (src / rel).exists() and (src / rel.parent).exists():
            items.append(rel)
    return items


def main() -> None:
    if not SRC.is_dir():
        sys.exit(f"원본 경로가 없습니다: {SRC}")

    if "--check" in sys.argv:
        if differs(SRC, DST):
            sys.exit("Skills 불일치: python scripts/sync_skills.py 실행 후 커밋하세요.")
        print("Skills 동기화 상태 정상")
        return

    extras = copy_only_items(SRC, DST)
    if extras and "--force" not in sys.argv:
        print("사본(.claude/skills/)에만 있는 항목이 있어 동기화를 중단합니다:")
        for rel in extras[:30]:
            print(f"  - {rel}")
        if len(extras) > 30:
            print(f"  ... 외 {len(extras) - 30}개")
        sys.exit(
            "필요한 스킬은 .agents/skills/(프로젝트 원본)로 옮긴 뒤 다시 실행하세요.\n"
            "지워도 되는 항목이면 --force로 실행하세요."
        )

    sync_in_place(SRC, DST)
    print(f"{SRC.relative_to(ROOT)} -> {DST.relative_to(ROOT)} 동기화 완료")


def sync_in_place(src: Path, dst: Path) -> None:
    """사본 폴더를 통째로 지우지 않고 바뀐 파일만 갱신한다.

    Windows에서 Claude Code가 .claude/skills/를 감시하면 폴더 삭제(rmtree)가
    PermissionError로 실패하므로, 원본에 없는 항목만 지우고 나머지는 덮어쓴다.
    """
    dst.mkdir(parents=True, exist_ok=True)
    # 원본에 없는 항목 삭제 (하위 항목부터)
    for path in sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if (src / path.relative_to(dst)).exists():
            continue
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    # 원본 항목 생성·갱신
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif not target.exists() or not filecmp.cmp(path, target, shallow=False):
            shutil.copy2(path, target)


if __name__ == "__main__":
    main()
