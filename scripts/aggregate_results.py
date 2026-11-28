"""main 병합 실행 기록 자동 집계 CLI 진입점 (SPEC.md US58, ADR-0006, Task T6.3).

기본 실행에서는 `results/runs/`가 git `main` 브랜치와 정확히 일치(미병합·미추적·누락 없음)하는지
검증한 뒤 `results/aggregated/`에 표·그림을 생성합니다.
병합 전 미리보기/테스트용 `--allow-unmerged`를 켜면 기본 공식 출력 폴더(`results/aggregated/`)로의
저장을 거부하고 별도 `--output-dir`만 허용합니다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from sr_pipeline.reporting.aggregate import export_report_assets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "aggregated"
DEFAULT_FAIR_CONFIG = REPO_ROOT / "configs" / "fair_comparison.yaml"
DEFAULT_EXPERIMENTS_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate merged run records into tables and figures.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Directory of merged run records (default: results/runs).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Aggregated outputs dir (default: results/aggregated).")
    parser.add_argument("--fair-config", type=Path, default=None, help="Path to fair_comparison.yaml.")
    parser.add_argument("--experiments-config", type=Path, default=None, help="Path to experiments.yaml.")
    parser.add_argument("--main-ref", type=str, default="main", help="Git ref representing merged records (default: main).")
    parser.add_argument(
        "--allow-unmerged",
        action="store_true",
        help="Skip main-branch verification for local preview (requires a non-default --output-dir).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional prepared data dir for case-study images.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint dir for case-study images.")
    return parser.parse_args(argv)


def _git_bytes(repo_root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "-c", "core.quotePath=false", *args],
            input=input_bytes,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git을 찾을 수 없어 main 병합 여부를 검증할 수 없습니다.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} 실패: {detail}") from error
    return result.stdout


def _git(repo_root: Path, *args: str) -> str:
    return _git_bytes(repo_root, *args).decode("utf-8", errors="replace").strip()


def _repo_relative(path: Path, repo_top: Path, *, label: str, main_ref: str) -> str:
    # 심볼릭 링크가 가리키는 바깥 대상이 아니라 저장소 안 경로 자체를 기준으로 판정한다.
    raw = Path(path)
    abs_path = raw if raw.is_absolute() else Path.cwd() / raw
    resolved_parent = abs_path.parent.resolve()
    candidate = resolved_parent / abs_path.name
    try:
        return candidate.relative_to(repo_top).as_posix()
    except ValueError as error:
        raise RuntimeError(
            f"{label}({candidate})가 git 저장소 밖에 있어 {main_ref} 병합 여부를 확인할 수 없습니다. "
            "미리보기용이면 --allow-unmerged와 별도 --output-dir을 지정하세요."
        ) from error


def _list_main_tree_entries(repo_top: Path, rel_paths: Sequence[str], main_ref: str) -> list[tuple[str, str, str]]:
    """`main_ref` 트리의 파일 목록 `(mode, sha, rel_path)`을 `-z`로 파싱합니다(비ASCII 파일명 인용 없음)."""
    raw = _git_bytes(repo_top, "ls-tree", "-r", "-z", main_ref, "--", *rel_paths)
    entries: list[tuple[str, str, str]] = []
    for record in raw.split(b"\x00"):
        if not record:
            continue
        meta_bytes, sep, path_bytes = record.partition(b"\t")
        if not sep:
            raise RuntimeError(f"git ls-tree 출력 형식을 해석할 수 없습니다: {record!r}")
        mode, obj_type, sha = meta_bytes.decode("ascii").split(" ")
        rel_path = path_bytes.decode("utf-8", errors="strict")
        if mode == "120000" or obj_type != "blob":
            raise RuntimeError(
                f"{main_ref}의 {rel_path}가 일반 파일(blob)이 아닙니다(mode={mode}, type={obj_type}). "
                "심볼릭 링크 등은 공식 집계에 사용할 수 없습니다."
            )
        entries.append((mode, sha, rel_path))
    return entries


def _verify_worktree_blobs(repo_top: Path, entries: Sequence[tuple[str, str, str]], main_ref: str) -> None:
    """작업 트리의 각 파일이 심볼릭 링크가 아니며 내용 해시가 `main_ref`의 blob과 일치하는지 확인합니다."""
    if not entries:
        return
    paths_for_hash: list[str] = []
    for _mode, _sha, rel_path in entries:
        fs_path = repo_top / Path(rel_path)
        if not fs_path.exists():
            raise RuntimeError(f"작업 트리에 {main_ref}의 파일이 없습니다: {rel_path}")
        if fs_path.is_symlink():
            raise RuntimeError(f"작업 트리의 {rel_path}가 심볼릭 링크입니다. 공식 집계에는 일반 파일만 허용됩니다.")
        paths_for_hash.append(rel_path)
    stdin_payload = "".join(f"{p}\n" for p in paths_for_hash).encode("utf-8")
    actual_shas = _git_bytes(repo_top, "hash-object", "--stdin-paths", input_bytes=stdin_payload).decode("ascii").splitlines()
    if len(actual_shas) != len(entries):
        raise RuntimeError("git hash-object 결과 수가 파일 목록과 일치하지 않습니다.")
    for (_mode, expected_sha, rel_path), actual_sha in zip(entries, actual_shas):
        if actual_sha.strip() != expected_sha:
            raise RuntimeError(
                f"{rel_path}의 실제 파일 내용이 {main_ref} 커밋과 다릅니다(skip-worktree 등으로 숨겨진 변경 포함)."
            )


def _read_main_blobs(repo_top: Path, shas: Sequence[str], main_ref: str) -> dict[str, bytes]:
    unique_shas = sorted(set(shas))
    if not unique_shas:
        return {}
    stdin_payload = "".join(f"{s}\n" for s in unique_shas).encode("ascii")
    raw = _git_bytes(repo_top, "cat-file", "--batch", input_bytes=stdin_payload)
    blobs: dict[str, bytes] = {}
    pos = 0
    for expected_sha in unique_shas:
        nl = raw.index(b"\n", pos)
        header = raw[pos:nl].decode("ascii")
        pos = nl + 1
        parts = header.split(" ")
        if len(parts) != 3 or parts[1] == "missing":
            raise RuntimeError(f"{main_ref}의 git 객체({expected_sha})를 읽을 수 없습니다: {header}")
        sha_got, _obj_type, size_str = parts
        size = int(size_str)
        blobs[sha_got] = raw[pos : pos + size]
        pos += size + 1
    return blobs


def _extract_snapshot(
    repo_top: Path,
    entries: Sequence[tuple[str, str, str]],
    main_ref: str,
    dest_root: Path,
) -> None:
    blobs = _read_main_blobs(repo_top, [sha for _m, sha, _p in entries], main_ref)
    for _mode, sha, rel_path in entries:
        target = dest_root / Path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[sha])


def _main_ref_run_ids(entries: Sequence[tuple[str, str, str]], rel_runs: str) -> set[str]:
    """`main_ref` 트리의 `rel_runs/<run_id>/...`에 실제로 병합된 실행 ID 목록을 돌려줍니다."""
    run_ids: set[str] = set()
    rel_prefix = Path(rel_runs)
    for _mode, _sha, rel_path in entries:
        path_obj = Path(rel_path)
        try:
            parts = path_obj.relative_to(rel_prefix).parts
        except ValueError:
            continue
        if len(parts) >= 2:
            run_ids.add(parts[0])
    return run_ids


def verify_runs_merged_to_main(
    runs_dir: Path,
    *,
    main_ref: str = "main",
    repo_root: Path = REPO_ROOT,
    config_paths: Sequence[Path] = (),
    snapshot_dir: Path | None = None,
) -> set[str]:
    """`runs_dir`과 설정 파일이 `main_ref` 브랜치와 100% 일치하는지 검증하고 병합된 `run_id` 집합을 돌려줍니다."""
    repo_top = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    rel_runs = _repo_relative(runs_dir, repo_top, label="실행 기록 폴더", main_ref=main_ref)
    rel_configs = [_repo_relative(path, repo_top, label="설정 파일", main_ref=main_ref) for path in config_paths]

    _git(repo_top, "rev-parse", "--verify", main_ref)
    for rel_path, kind in [(rel_runs, "실행 기록"), *[(p, "설정 파일") for p in rel_configs]]:
        # `--ignored`로 `.git/info/exclude`·`.gitignore`에 숨긴 미병합 파일까지 모두 탐지한다.
        dirty = _git_bytes(repo_top, "status", "--porcelain", "-z", "--untracked-files=all", "--ignored", "--", rel_path)
        if dirty:
            dirty_text = dirty.replace(b"\x00", b"\n").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{rel_path}에 커밋되지 않았거나 추적/제외(.gitignore·.git/info/exclude)된 {kind}이 있습니다. "
                f"커밋 및 {main_ref} 병합 후 집계하세요:\n{dirty_text}"
            )
        diff = _git_bytes(repo_top, "diff", "--name-status", "-z", main_ref, "--", rel_path)
        if diff:
            diff_text = diff.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{rel_path}가 {main_ref} 브랜치에 병합된 {kind}과 다릅니다"
                f"(미병합 변경이 있거나 {main_ref} 최신 병합 기록이 빠져 있습니다):\n{diff_text}"
            )

    all_paths = [rel_runs, *rel_configs]
    entries = _list_main_tree_entries(repo_top, all_paths, main_ref)
    entry_paths = {p for _m, _s, p in entries}
    for rel_cfg in rel_configs:
        if rel_cfg not in entry_paths:
            raise RuntimeError(f"{main_ref} 브랜치에 필수 설정 파일이 없습니다: {rel_cfg}")
    _verify_worktree_blobs(repo_top, entries, main_ref)
    if snapshot_dir is not None:
        _extract_snapshot(repo_top, entries, main_ref, snapshot_dir)
    return _main_ref_run_ids(entries, rel_runs)


def _check_output_target(
    output_dir: Path,
    runs_dir: Path,
    fair_config: Path,
    experiments_config: Path,
    allow_unmerged: bool,
    main_ref: str,
) -> None:
    """공식 출력 폴더(`results/aggregated/`)에는 main 병합 검증과 기본 설정을 거친 집계만 허용한다."""
    default_out = DEFAULT_OUTPUT_DIR.resolve()
    resolved_out = Path(output_dir).resolve()
    is_default_target = resolved_out == default_out or default_out in resolved_out.parents
    if not is_default_target:
        return
    if allow_unmerged or main_ref != "main":
        raise ValueError(
            "공식 집계 폴더(results/aggregated/)에는 main 기준 검증을 통과한 집계만 쓸 수 있습니다"
            "(--allow-unmerged 또는 --main-ref 변경 시 별도 --output-dir을 지정하세요)."
        )
    if (
        Path(runs_dir).resolve() != DEFAULT_RUNS_DIR.resolve()
        or Path(fair_config).resolve() != DEFAULT_FAIR_CONFIG.resolve()
        or Path(experiments_config).resolve() != DEFAULT_EXPERIMENTS_CONFIG.resolve()
    ):
        raise ValueError(
            "공식 집계 폴더(results/aggregated/)에는 기본 results/runs 및 configs/ 설정을 쓴 집계만 저장할 수 있습니다. "
            "다른 경로·설정을 쓰려면 별도 --output-dir을 지정하세요."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir = args.runs_dir if args.runs_dir is not None else DEFAULT_RUNS_DIR
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    fair_config = args.fair_config if args.fair_config is not None else DEFAULT_FAIR_CONFIG
    experiments_config = args.experiments_config if args.experiments_config is not None else DEFAULT_EXPERIMENTS_CONFIG

    _check_output_target(output_dir, runs_dir, fair_config, experiments_config, args.allow_unmerged, args.main_ref)
    if args.allow_unmerged:
        manifest = export_report_assets(
            runs_dir,
            output_dir,
            fair_config_path=fair_config,
            experiments_config_path=experiments_config,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        snapshot_root = Path(tempfile.mkdtemp(prefix="aggregate_main_snapshot_"))
        try:
            allowed_run_ids = verify_runs_merged_to_main(
                runs_dir,
                main_ref=args.main_ref,
                repo_root=REPO_ROOT,
                config_paths=(fair_config, experiments_config),
                snapshot_dir=snapshot_root,
            )
            repo_top = Path(_git(REPO_ROOT, "rev-parse", "--show-toplevel")).resolve()
            rel_runs = _repo_relative(runs_dir, repo_top, label="실행 기록 폴더", main_ref=args.main_ref)
            rel_fair = _repo_relative(fair_config, repo_top, label="설정 파일", main_ref=args.main_ref)
            rel_exp = _repo_relative(experiments_config, repo_top, label="설정 파일", main_ref=args.main_ref)
            manifest = export_report_assets(
                snapshot_root / Path(rel_runs),
                output_dir,
                fair_config_path=snapshot_root / Path(rel_fair),
                experiments_config_path=snapshot_root / Path(rel_exp),
                data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir,
                allowed_run_ids=allowed_run_ids,
            )
        finally:
            shutil.rmtree(snapshot_root, ignore_errors=True)
    print(f"aggregated {len(manifest['run_ids'])} runs -> {len(manifest['tables'])} tables, {len(manifest['figures'])} figures")


if __name__ == "__main__":
    main()
