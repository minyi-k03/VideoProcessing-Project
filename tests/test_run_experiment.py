"""`scripts/run_experiment.py` 바깥 동작 테스트 (T3.8, SPEC.md US2, 7, 31, 38–39).

구성 + 시드 + 경로 인자 → 실행 기록 폴더. 합성 패치 풀·테스트셋과 테스트 코드 안 점검용 fair 파일만 쓴다
(`pilot_determined`가 `null`인 실제 `configs/fair_comparison.yaml`로는 실행 조건을 확정할 수 없다, T4.2 전).
Colab식(`--data-archive`)과 Kaggle식(`--data-dir`에 사전 해제된 폴더)이 같은 결과를 내는지 본다.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from experiment_fixtures import DATASETS, POOL_ID, REPO_ROOT, args_for, git, only_run_dir
from sr_pipeline.data.prepare import pack_archive
from sr_pipeline.engine.runtime_data import extract_data_archive, sha256_of

pytestmark = pytest.mark.usefixtures("restore_global_random_state")  # 학습이 set_model_seed를 부른다


@pytest.fixture(scope="module")
def archive(prepared_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return pack_archive(prepared_data, tmp_path_factory.mktemp("drive") / "sr_data.zip")


def test_drive_style_archive_run_produces_run_record(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(args_for(tmp_path, fair_path, archive=archive))

    run_dir = only_run_dir(tmp_path / "records")
    assert {path.name for path in run_dir.iterdir()} == {"summary.csv", "per_image.csv", "train_log.json", "images"}
    summary = pd.read_csv(run_dir / "summary.csv")
    assert list(summary["dataset"]) == list(DATASETS)
    assert set(summary["config_id"]) == {"baseline"} and set(summary["seed"]) == {0}
    assert set(summary["status"]) == {"complete"} and set(summary["patch_pool_id"]) == {POOL_ID}
    assert len(pd.read_csv(run_dir / "per_image.csv")) == 2 * len(DATASETS)
    assert (tmp_path / "ckpt" / run_dir.name / "weights.pt").is_file()
    assert (tmp_path / "local_data" / "patch_pool" / "metadata.json").is_file()  # 런타임 로컬로 풀렸다

    output = capsys.readouterr()
    assert run_dir.name in output.out
    assert str(tmp_path) not in output.out + output.err  # 입력·저장 경로는 출력하지 않는다(Public 저장소)
    assert str(tmp_path) not in (run_dir / "train_log.json").read_text(encoding="utf-8")


def test_kaggle_style_prepared_input_folder_gives_the_same_record_and_is_not_modified(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    kaggle_input = tmp_path / "input"
    extract_data_archive(archive, kaggle_input / "datasets" / "someone" / "slug")  # 자동 해제돼 한 단계 감싸진 Dataset
    before = sorted(path.relative_to(kaggle_input) for path in kaggle_input.rglob("*"))

    cli.main(args_for(tmp_path, fair_path, data_dir=kaggle_input, output_dir=tmp_path / "kaggle_records"))

    assert sorted(path.relative_to(kaggle_input) for path in kaggle_input.rglob("*")) == before
    cli.main(args_for(tmp_path, fair_path, archive=archive, output_dir=tmp_path / "drive_records",
                      checkpoint_dir=tmp_path / "ckpt2"))
    kaggle_summary = pd.read_csv(only_run_dir(tmp_path / "kaggle_records") / "summary.csv")
    drive_summary = pd.read_csv(only_run_dir(tmp_path / "drive_records") / "summary.csv")
    # 같은 코드·구성·시드·데이터면 입력 경로와 상관없이 같은 학습 결과(시간 필드는 제외)다.
    timing = ["train_minutes", "infer_ms_net", "infer_ms_total"]
    pd.testing.assert_frame_equal(kaggle_summary.drop(columns=timing), drive_summary.drop(columns=timing))


def test_bicubic_runs_without_training_and_without_checkpoints(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    cli.main(args_for(tmp_path, fair_path, config="bicubic", archive=archive))

    run_dir = only_run_dir(tmp_path / "records")
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["config_id"]) == {"bicubic"} and set(summary["status"]) == {"complete"}
    assert (summary["params"] == 0).all() and (summary["train_minutes"] == 0).all()
    assert not (run_dir / "train_log.json").exists()
    assert not (tmp_path / "ckpt").exists() or not any((tmp_path / "ckpt").iterdir())


def test_same_run_is_not_overwritten(cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path) -> None:
    cli.main(args_for(tmp_path, fair_path, archive=archive))
    run_dir = only_run_dir(tmp_path / "records")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    with pytest.raises(FileExistsError):
        cli.main(args_for(tmp_path, fair_path, archive=archive))
    assert {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()} == before


def test_undetermined_fair_comparison_is_rejected_before_touching_data_or_outputs(
    cli: ModuleType, tmp_path: Path, archive: Path
) -> None:
    """실제 `configs/fair_comparison.yaml`은 T4.2 전이라 `null`이 남아 있다: 해제·학습 전에 멈춘다."""
    argv = args_for(tmp_path, REPO_ROOT / "configs" / "fair_comparison.yaml", archive=archive)

    with pytest.raises(ValueError, match="null"):
        cli.main(argv)
    for name in ("local_data", "records", "ckpt"):
        assert not (tmp_path / name).exists()


@pytest.mark.parametrize("override", [{"config": "not_a_config"}, {"seed": 7}])
def test_invalid_config_or_seed_is_rejected_before_touching_data(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, override: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        cli.main(args_for(tmp_path, fair_path, archive=archive, **override))
    assert not (tmp_path / "local_data").exists() and not (tmp_path / "records").exists()


def test_uncommitted_code_changes_are_rejected_before_unpacking_unless_allowed(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, code_repo: Path
) -> None:
    (code_repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="커밋되지 않은 변경"):
        cli.main(args_for(tmp_path, fair_path, archive=archive))
    assert not (tmp_path / "local_data").exists()

    with pytest.warns(UserWarning, match="커밋되지 않은 변경"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--allow-dirty"])
    assert only_run_dir(tmp_path / "records").name.startswith(git(code_repo, "rev-parse", "HEAD") + "-dirty-")


def test_wrong_archive_sha256_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    with pytest.raises(ValueError, match="sha256"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--archive-sha256", "0" * 64])
    assert not (tmp_path / "local_data").exists()

    cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--archive-sha256", sha256_of(archive)])
    assert only_run_dir(tmp_path / "records")


@pytest.mark.parametrize(
    ("config", "resume_run_id", "message"),
    [("baseline", "other-baseline-s0", "resume_run_id"), ("bicubic", "other-bicubic-s0", "학습하지 않는 구성")],
)
def test_invalid_resume_request_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, config: str, resume_run_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cli.main(args_for(tmp_path, fair_path, config=config, archive=archive) + ["--resume-run-id", resume_run_id])
    assert not (tmp_path / "local_data").exists() and not (tmp_path / "records").exists()


def test_dirty_run_cannot_be_resumed_and_is_rejected_before_unpacking(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, code_repo: Path
) -> None:
    (code_repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    dirty_run_id = f"{git(code_repo, 'rev-parse', 'HEAD')}-dirty-baseline-s0"

    with pytest.warns(UserWarning), pytest.raises(ValueError, match="재개할 수 없습니다"):
        cli.main(args_for(tmp_path, fair_path, archive=archive) + ["--allow-dirty", "--resume-run-id", dirty_run_id])
    assert not (tmp_path / "local_data").exists()


def test_default_config_paths_do_not_depend_on_the_working_directory(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--experiments-config`를 주지 않고 저장소 밖에서 실행해도 저장소의 `configs/experiments.yaml`을 쓴다."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cli.main(args_for(tmp_path, fair_path, archive=archive))

    assert only_run_dir(tmp_path / "records")
    defaults = cli.parse_args(["--config", "baseline", "--seed", "0", "--data-dir", "d", "--checkpoint-dir", "c"])
    assert defaults.output_dir == REPO_ROOT / "results" / "runs"
    assert defaults.fair_config == REPO_ROOT / "configs" / "fair_comparison.yaml"


def test_help_lists_the_path_arguments(cli: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--data-dir", "--data-archive", "--checkpoint-dir", "--output-dir", "--resume-run-id", "--allow-dirty"):
        assert option in help_text


def test_default_results_runs_dir_rejects_non_default_configs(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    """저장소의 기본 `results/runs/` 경로에는 점검용 `--fair-config`나 별도 `--experiments-config`를 쓸 수 없다."""
    argv = [
        "--config", "baseline", "--seed", "0", "--device", "cpu",
        "--fair-config", str(fair_path),
        "--data-dir", str(tmp_path / "local_data"),
        "--data-archive", str(archive),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
    ]
    with pytest.raises(ValueError, match="results/runs/"):
        cli.main(argv)
    assert not (tmp_path / "local_data").exists()

    custom_exp = tmp_path / "experiments.yaml"
    custom_exp.write_text((REPO_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="results/runs/"):
        cli.main([
            "--config", "baseline", "--seed", "0", "--device", "cpu",
            "--experiments-config", str(custom_exp),
            "--output-dir", str(REPO_ROOT / "results" / "runs" / "sub"),
            "--data-dir", str(tmp_path / "local_data"),
            "--data-archive", str(archive),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])


def test_bicubic_rejects_mismatched_patch_pool_id(
    cli: ModuleType, tmp_path: Path, fair_path: Path, archive: Path
) -> None:
    """학습하지 않는 Bicubic 하한도 입력 데이터의 실제 `patch_pool_id`가 설정과 다르면 거부한다."""
    data_dir = tmp_path / "prepared"
    extract_data_archive(archive, data_dir)
    meta_path = data_dir / "patch_pool" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["patch_pool_id"] = "other_pool"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="patch_pool_id"):
        cli.main(args_for(tmp_path, fair_path, config="bicubic", data_dir=data_dir))
    assert not (tmp_path / "records").exists()
