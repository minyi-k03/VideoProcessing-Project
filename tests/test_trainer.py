"""정식 트레이너(T3.7) 바깥 동작 테스트 (SPEC.md US16, 19–20, 31–37).

작은 합성 패치 풀·로컬 CPU만 쓴다. 여기 쓰인 학습 설정 값은 테스트 코드 안 합성 값이며,
`pilot_determined`가 `null`인 실제 `fair_comparison.yaml`로는 실행 조건을 확정할 수 없다(T4.2 전).
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from sr_pipeline.config.schema import ResolvedRunConfig, Switches, resolve_run_config
from sr_pipeline.engine import trainer
from sr_pipeline.engine.trainer import (
    STATE_FILENAME,
    TRAIN_LOG_FILENAME,
    WEIGHTS_FILENAME,
    build_model_and_loss,
    build_run_id,
    check_git_clean,
    train_and_evaluate,
    train_run,
)

pytestmark = pytest.mark.usefixtures("restore_global_random_state")  # train_run이 set_model_seed를 부른다

REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE = 4
PATCH_HR = 16
POOL_SIZE = 12
POOL_ID = "synthetic_pool"


# ---- 실행 ID와 커밋 확인 (US34-35) ----


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in ("src/module.py", "scripts/run.py", "configs/rules.yaml"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x = 1\n", encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "add", ".")
    _git(root, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "init")
    return root


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없는 환경")


def test_run_id_is_commit_config_seed() -> None:
    assert build_run_id("abc123", "AB", 2) == "abc123-AB-s2"


@needs_git
def test_check_git_clean_returns_head_commit(repo: Path) -> None:
    assert check_git_clean(repo) == _git(repo, "rev-parse", "HEAD")


@needs_git
def test_check_git_clean_rejects_uncommitted_change_unless_allowed(repo: Path) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        check_git_clean(repo)
    with pytest.warns(UserWarning, match="커밋되지 않은 변경"):
        assert check_git_clean(repo, allow_dirty=True) == _git(repo, "rev-parse", "HEAD") + "-dirty"


# ---- 합성 데이터와 실행 조건 ----

COMMIT = "c0ffee"
SYNTHETIC_PILOT_DETERMINED: dict[str, Any] = {
    "patch_pool_id": POOL_ID,
    "patch_pool_size": POOL_SIZE,
    "patch_size_hr": PATCH_HR,
    "patches_per_epoch": 8,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "scheduler": "none",
    "augmentation": "rot90x4_hflip",
    "edge_loss_lambda": 0.1,
    "edge_top_q": 10,
    "representative_images": {
        name: {"median": "a", "worst": "b"} for name in ("Set5", "Set14", "BSD100", "Urban100")
    },
}


def _write_pool(data_dir: Path, seed: int = 123) -> None:
    """`data.prepare.create_patch_pool`과 같은 파일 구조의 작은 합성 패치 풀을 만든다."""
    rng = np.random.default_rng(seed)
    pool_dir = data_dir / "patch_pool"
    pool_dir.mkdir(parents=True)
    np.save(pool_dir / "hr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR, PATCH_HR, 3), dtype=np.uint8))
    np.save(pool_dir / "lr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR // SCALE, PATCH_HR // SCALE, 3), dtype=np.uint8))
    metadata = {"patch_pool_id": POOL_ID, "scale": SCALE, "num_patches": POOL_SIZE}
    (pool_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    _write_pool(path)
    return path


def _run_config(
    tmp_path: Path,
    config_id: str = "baseline",
    seed: int = 0,
    *,
    pilot: dict[str, Any] | None = None,
    max_hours: float | None = None,
) -> ResolvedRunConfig:
    """실제 `experiments.yaml`과 테스트 전용 합성 값으로 채운 fair 파일로 실행 조건을 확정한다."""
    fair = yaml.safe_load((REPO_ROOT / "configs" / "fair_comparison.yaml").read_text(encoding="utf-8"))
    fair["pilot_determined"] = {**copy.deepcopy(SYNTHETIC_PILOT_DETERMINED), **(pilot or {})}
    if max_hours is not None:
        fair["fixed_rules"]["max_hours"] = max_hours
    fair_path = tmp_path / "fair_comparison.yaml"
    fair_path.write_text(yaml.safe_dump(fair, allow_unicode=True), encoding="utf-8")
    return resolve_run_config(fair_path, REPO_ROOT / "configs" / "experiments.yaml", config_id, seed)


def _train(tmp_path: Path, data_dir: Path, run_config: ResolvedRunConfig, label: str = "run", **kwargs: Any):
    """`<tmp>/<label>/{checkpoints,records}`에 쓰며 `train_run`을 부른다."""
    return train_run(
        run_config,
        data_dir,
        tmp_path / label / "records",
        tmp_path / label / "checkpoints",
        commit=COMMIT,
        **kwargs,
    )


# ---- 학습 완주와 기록 (US31, 36-37) ----


def test_train_run_completes_and_writes_checkpoints_and_log(tmp_path: Path, data_dir: Path) -> None:
    result = _train(tmp_path, data_dir, _run_config(tmp_path))

    run_id = build_run_id(COMMIT, "baseline", 0)
    assert result.run_id == run_id
    assert result.status == "complete"
    assert result.epochs_done == result.epochs_total == 3
    assert len(result.epoch_losses) == 3 and all(np.isfinite(result.epoch_losses))
    assert result.train_minutes >= 0

    checkpoint_dir = tmp_path / "run" / "checkpoints" / run_id
    assert (checkpoint_dir / STATE_FILENAME).is_file()
    weights = torch.load(checkpoint_dir / WEIGHTS_FILENAME, weights_only=True)
    assert set(weights) == set(result.model.state_dict())

    log = json.loads((tmp_path / "run" / "records" / run_id / TRAIN_LOG_FILENAME).read_text(encoding="utf-8"))
    assert log == {
        "run_id": run_id,
        "commit": COMMIT,
        "config_id": "baseline",
        "seed": 0,
        "patch_pool_id": POOL_ID,
        "switches": {"A": False, "B": False, "C": False, "Bp": False},
        "params": 37_200,
        "status": "complete",
        "epochs_done": 3,
        "epochs_total": 3,
        "train_minutes": result.train_minutes,
        "epoch_losses": result.epoch_losses,
    }
    assert str(tmp_path) not in json.dumps(log)  # Public 저장소: 경로는 기록하지 않는다


# ---- 재현성 (US32-33) ----


def _weights(result) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in result.model.state_dict().items()}


def _assert_same_weights(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> None:
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_same_config_and_seed_reproduce_weights_and_losses_and_other_seed_differs(
    tmp_path: Path, data_dir: Path
) -> None:
    first = _train(tmp_path, data_dir, _run_config(tmp_path), "first")
    second = _train(tmp_path, data_dir, _run_config(tmp_path), "second")
    other = _train(tmp_path, data_dir, _run_config(tmp_path, seed=1), "other")

    assert first.epoch_losses == second.epoch_losses
    _assert_same_weights(_weights(first), _weights(second))
    assert first.epoch_losses != other.epoch_losses


# ---- 시계와 재개 (US36) ----


class TickClock:
    """호출할 때마다 `tick`초씩 흐르는 가짜 시계. `fail_at_call`번째 호출에서 런타임이 끊기듯 예외를 던진다.

    트레이너는 에폭마다 시작·끝에 시계를 한 번씩만 읽는다(에폭 학습 시간 = 끝 − 시작).
    """

    def __init__(self, tick: float = 1.0, fail_at_call: int | None = None) -> None:
        self.tick, self.fail_at_call, self.calls, self.now = tick, fail_at_call, 0, 0.0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.fail_at_call:
            raise ConnectionError("런타임이 끊겼습니다")
        self.now += self.tick
        return self.now


def test_resume_after_runtime_cut_matches_uninterrupted_run(tmp_path: Path, data_dir: Path) -> None:
    run_config = _run_config(tmp_path)
    full = _train(tmp_path, data_dir, run_config, "cut_and_full_share_nothing", clock=TickClock())

    # 3에폭 중 2에폭을 끝내고 3번째 에폭을 시작하려는 순간(시계 5번째 호출) 런타임이 끊긴다.
    with pytest.raises(ConnectionError):
        _train(tmp_path, data_dir, run_config, "cut", clock=TickClock(fail_at_call=5))
    resumed = _train(tmp_path, data_dir, run_config, "cut", resume=True, clock=TickClock())

    assert resumed.status == "complete"
    assert resumed.epoch_losses == full.epoch_losses
    _assert_same_weights(_weights(resumed), _weights(full))
    # 끊기기 전 두 에폭의 학습 시간도 이어서 센다(에폭당 1초 × 3).
    assert resumed.train_minutes == pytest.approx(3 / 60)


# ---- 3시간 상한 (US16, 20) ----

HOUR = 3600.0


def test_time_cap_stops_before_next_epoch_and_records_incomplete(tmp_path: Path, data_dir: Path) -> None:
    # 에폭당 1시간, 상한 1.5시간: 에폭 2개를 끝낸 뒤(누적 2시간) 3번째 에폭을 시작하기 전에 멈춘다.
    run_config = _run_config(tmp_path, max_hours=1.5)
    result = _train(tmp_path, data_dir, run_config, clock=TickClock(tick=HOUR))

    assert result.status == "incomplete"
    assert (result.epochs_done, result.epochs_total) == (2, 3)
    assert len(result.epoch_losses) == 2
    assert result.train_minutes == pytest.approx(120.0)
    log = json.loads(
        (tmp_path / "run" / "records" / result.run_id / TRAIN_LOG_FILENAME).read_text(encoding="utf-8")
    )
    assert (log["status"], log["epochs_done"], log["epochs_total"]) == ("incomplete", 2, 3)
    assert log["train_minutes"] == pytest.approx(120.0)
    assert (tmp_path / "run" / "checkpoints" / result.run_id / WEIGHTS_FILENAME).is_file()


def test_time_cap_exceeded_during_the_last_epoch_is_still_complete(tmp_path: Path, data_dir: Path) -> None:
    # 상한 2.5시간: 마지막 에폭 도중 3시간이 되어도 모든 에폭을 끝냈으므로 complete다.
    result = _train(tmp_path, data_dir, _run_config(tmp_path, max_hours=2.5), clock=TickClock(tick=HOUR))

    assert result.status == "complete"
    assert result.epochs_done == 3
    assert result.train_minutes == pytest.approx(180.0)


def test_resuming_a_capped_run_trains_no_further_and_stays_incomplete(tmp_path: Path, data_dir: Path) -> None:
    run_config = _run_config(tmp_path, max_hours=1.5)
    first = _train(tmp_path, data_dir, run_config, clock=TickClock(tick=HOUR))

    clock = TickClock(tick=HOUR)
    again = _train(tmp_path, data_dir, run_config, resume=True, clock=clock)

    assert clock.calls == 0
    assert again.status == "incomplete"
    assert again.epochs_done == 2
    assert again.epoch_losses == first.epoch_losses
    assert again.train_minutes == pytest.approx(120.0)
    _assert_same_weights(_weights(again), _weights(first))


# ---- 시작·재개 거부 (US35-36) ----


def test_fresh_start_is_refused_when_the_run_already_has_a_checkpoint(tmp_path: Path, data_dir: Path) -> None:
    run_config = _run_config(tmp_path)
    first = _train(tmp_path, data_dir, run_config)

    with pytest.raises(FileExistsError):
        _train(tmp_path, data_dir, run_config)
    # 거부된 시도가 이전 실행의 기록을 건드리지 않는다.
    log_path = tmp_path / "run" / "records" / first.run_id / TRAIN_LOG_FILENAME
    assert json.loads(log_path.read_text(encoding="utf-8"))["epoch_losses"] == first.epoch_losses


def test_resume_without_a_checkpoint_is_refused(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _train(tmp_path, data_dir, _run_config(tmp_path), resume=True)


def test_resume_is_refused_when_training_conditions_changed(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(ConnectionError):
        _train(tmp_path, data_dir, _run_config(tmp_path), clock=TickClock(fail_at_call=5))

    changed = _run_config(tmp_path, pilot={"learning_rate": 5e-4})
    with pytest.raises(ValueError, match="학습 설정"):
        _train(tmp_path, data_dir, changed, resume=True)


def test_resume_is_refused_for_a_checkpoint_of_another_run(tmp_path: Path, data_dir: Path) -> None:
    seed0 = _train(tmp_path, data_dir, _run_config(tmp_path, seed=0))
    seed1_id = build_run_id(COMMIT, "baseline", 1)
    checkpoints = tmp_path / "run" / "checkpoints"
    shutil.copytree(checkpoints / seed0.run_id, checkpoints / seed1_id)  # 다른 실행의 체크포인트를 잘못 가져온 상황

    with pytest.raises(ValueError, match="run_id"):
        _train(tmp_path, data_dir, _run_config(tmp_path, seed=1), resume=True)


# ---- 구성별 스위치 배선 (US28-29) ----

ONE_EPOCH = {"epochs": 1}


@pytest.mark.parametrize(
    ("config_id", "params"),
    [("baseline", 37_200), ("A", 37_200), ("B", 24_752), ("AB", 24_752), ("C", 37_200), ("Bp", 37_200)],
)
def test_every_trainable_config_trains_and_reports_its_parameter_count(
    tmp_path: Path, data_dir: Path, config_id: str, params: int
) -> None:
    result = _train(tmp_path, data_dir, _run_config(tmp_path, config_id, pilot=ONE_EPOCH))

    assert result.status == "complete"
    assert result.params == params
    assert len(result.epoch_losses) == 1 and np.isfinite(result.epoch_losses[0])


def test_all_switches_off_is_the_baseline_and_edge_loss_lambda_zero_matches_it(tmp_path: Path, data_dir: Path) -> None:
    baseline = _train(tmp_path, data_dir, _run_config(tmp_path, "baseline"), "baseline")
    edge_off = _train(tmp_path, data_dir, _run_config(tmp_path, "A", pilot={"edge_loss_lambda": 0.0}), "lambda0")
    edge_on = _train(tmp_path, data_dir, _run_config(tmp_path, "A"), "lambda")

    assert edge_off.epoch_losses == baseline.epoch_losses
    _assert_same_weights(_weights(edge_off), _weights(baseline))
    assert edge_on.epoch_losses != baseline.epoch_losses  # λ는 fair_comparison의 edge_loss_lambda에서 읽는다


def test_y_loss_control_changes_only_the_loss(tmp_path: Path, data_dir: Path) -> None:
    baseline = _train(tmp_path, data_dir, _run_config(tmp_path, "baseline", pilot=ONE_EPOCH), "baseline")
    control = _train(tmp_path, data_dir, _run_config(tmp_path, "Bp", pilot=ONE_EPOCH), "control")

    assert control.params == baseline.params
    assert control.epoch_losses != baseline.epoch_losses


def test_y_channel_and_y_loss_control_cannot_both_be_on() -> None:
    fair = {"pilot_determined": {"edge_loss_lambda": 0.1}}
    with pytest.raises(ValueError, match="Bp"):
        build_model_and_loss(Switches(B=True, Bp=True), fair)


def test_a_config_without_training_is_refused_before_anything_is_written(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(ValueError, match="학습"):
        _train(tmp_path, data_dir, _run_config(tmp_path, "bicubic"))
    assert not (tmp_path / "run").exists()


# ---- 잘못된 설정은 쓰기 전에 거부 (US13, 15) ----


@pytest.mark.parametrize(
    "pilot",
    [
        {"optimizer": "sgd"},
        {"scheduler": "cosine"},
        {"epochs": 0},
        {"batch_size": 1.5},
        {"patches_per_epoch": True},
        {"learning_rate": -1e-3},
        {"learning_rate": float("inf")},
        {"augmentation": "mixup"},
        {"patch_pool_id": "other_pool"},
        {"patch_pool_size": POOL_SIZE + 1},
    ],
    ids=lambda pilot: next(iter(pilot)),
)
def test_invalid_settings_or_mismatched_pool_fail_before_anything_is_written(
    tmp_path: Path, data_dir: Path, pilot: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        _train(tmp_path, data_dir, _run_config(tmp_path, pilot=pilot))
    assert not (tmp_path / "run").exists()


def test_pool_whose_scale_differs_from_the_model_is_rejected(tmp_path: Path) -> None:
    pool_dir = tmp_path / "x2_data" / "patch_pool"
    pool_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(pool_dir / "hr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR, PATCH_HR, 3), dtype=np.uint8))
    np.save(pool_dir / "lr.npy", rng.integers(0, 256, (POOL_SIZE, PATCH_HR // 2, PATCH_HR // 2, 3), dtype=np.uint8))
    (pool_dir / "metadata.json").write_text(json.dumps({"patch_pool_id": POOL_ID, "scale": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="배율"):
        _train(tmp_path, tmp_path / "x2_data", _run_config(tmp_path))
    assert not (tmp_path / "run").exists()


# ---- 학습부터 평가까지 한 번에 (US31, 34-36) ----
# 실제 평가(`evaluate_model`)는 T3.6 몫이라 호출 경계만 확인한다.


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model, run_meta, data_dir, output_dir):
        self.calls.append({"model": model, "run_meta": run_meta, "data_dir": data_dir, "output_dir": output_dir})
        return [], []


@pytest.fixture
def evaluator(monkeypatch: pytest.MonkeyPatch) -> FakeEvaluator:
    fake = FakeEvaluator()
    monkeypatch.setattr(trainer, "evaluate_model", fake)
    return fake


@needs_git
def test_train_and_evaluate_trains_then_hands_the_model_and_run_meta_to_the_evaluator(
    tmp_path: Path, data_dir: Path, repo: Path, evaluator: FakeEvaluator
) -> None:
    commit = _git(repo, "rev-parse", "HEAD")
    run_dir = train_and_evaluate(
        _run_config(tmp_path, "B", pilot=ONE_EPOCH),
        data_dir,
        tmp_path / "records",
        tmp_path / "checkpoints",
        repo_root=repo,
    )

    run_id = build_run_id(commit, "B", 0)
    assert run_dir == tmp_path / "records" / run_id
    (call,) = evaluator.calls
    assert call["output_dir"] == run_dir and call["data_dir"] == data_dir
    assert call["model"].use_y_channel and not call["model"].training
    assert call["run_meta"] == {
        "run_id": run_id,
        "commit": commit,
        "config_id": "B",
        "seed": 0,
        "patch_pool_id": POOL_ID,
        "switches": {"A": False, "B": True, "C": False, "Bp": False},
        "params": 24_752,
        "train_minutes": call["run_meta"]["train_minutes"],
        "status": "complete",
    }
    assert (run_dir / TRAIN_LOG_FILENAME).is_file()


@needs_git
def test_train_and_evaluate_refuses_a_dirty_tree_before_touching_data(
    tmp_path: Path, repo: Path, evaluator: FakeEvaluator
) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        train_and_evaluate(
            _run_config(tmp_path), tmp_path / "no_such_data", tmp_path / "records", tmp_path / "checkpoints", repo_root=repo
        )
    assert evaluator.calls == [] and not (tmp_path / "records").exists()


@needs_git
def test_train_and_evaluate_allow_dirty_marks_the_run_id(
    tmp_path: Path, data_dir: Path, repo: Path, evaluator: FakeEvaluator
) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="커밋되지 않은 변경"):
        run_dir = train_and_evaluate(
            _run_config(tmp_path, pilot=ONE_EPOCH),
            data_dir,
            tmp_path / "records",
            tmp_path / "checkpoints",
            allow_dirty=True,
            repo_root=repo,
        )
    assert run_dir.name == build_run_id(_git(repo, "rev-parse", "HEAD") + "-dirty", "baseline", 0)


@needs_git
def test_train_and_evaluate_skips_training_for_a_config_without_training(
    tmp_path: Path, data_dir: Path, repo: Path, evaluator: FakeEvaluator
) -> None:
    run_dir = train_and_evaluate(
        _run_config(tmp_path, "bicubic"), data_dir, tmp_path / "records", tmp_path / "checkpoints", repo_root=repo
    )

    (call,) = evaluator.calls
    assert call["model"] is None
    assert (call["run_meta"]["status"], call["run_meta"]["train_minutes"], call["run_meta"]["params"]) == ("complete", 0.0, 0)
    assert not (run_dir / TRAIN_LOG_FILENAME).exists() and not (tmp_path / "checkpoints").exists()


@needs_git
def test_train_and_evaluate_resume_run_id_must_match_and_continues_the_run(
    tmp_path: Path, data_dir: Path, repo: Path, evaluator: FakeEvaluator
) -> None:
    commit = _git(repo, "rev-parse", "HEAD")
    run_config = _run_config(tmp_path)
    with pytest.raises(ConnectionError):  # 런타임이 끊겨 2에폭까지 저장된 상황
        train_run(
            run_config,
            data_dir,
            tmp_path / "records",
            tmp_path / "checkpoints",
            commit=commit,
            clock=TickClock(fail_at_call=5),
        )
    args = (run_config, data_dir, tmp_path / "records", tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="resume_run_id"):
        train_and_evaluate(*args, resume_run_id=build_run_id(commit, "baseline", 1), repo_root=repo)
    assert evaluator.calls == []

    train_and_evaluate(*args, resume_run_id=build_run_id(commit, "baseline", 0), repo_root=repo)
    (call,) = evaluator.calls
    assert call["run_meta"]["status"] == "complete"
    uninterrupted = _train(tmp_path, data_dir, run_config, "uninterrupted")
    _assert_same_weights(call["model"].state_dict(), _weights(uninterrupted))
