"""공정 비교 조건 검증 및 시드 기반 샘플러 테스트 (SPEC.md Testing Decisions §1)."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from sr_pipeline.config.schema import (
    SWITCH_NAMES,
    Switches,
    find_undetermined,
    load_experiment_config,
    load_fair_comparison,
    resolve_run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_FAIR = REPO_ROOT / "configs" / "fair_comparison.yaml"
REAL_EXPERIMENTS = REPO_ROOT / "configs" / "experiments.yaml"

# 테스트 전용 합성 값. 실제 확정 값은 T4.2에서 fair_comparison.yaml에 고정한다.
SYNTHETIC_PILOT_DETERMINED: dict[str, Any] = {
    "patch_pool_id": "synthetic_pool",
    "patch_pool_size": 8,
    "patch_size_hr": 8,
    "patches_per_epoch": 4,
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "scheduler": "none",
    "augmentation": "flip_rot",
    "edge_loss_lambda": 0.1,
    "edge_top_q": 10,
    "representative_images": {
        "Set5": {"median": "a", "worst": "b"},
        "Set14": {"median": "c", "worst": "d"},
    },
}


def _write_yaml(path: Path, data: Any) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _synthetic_fair(tmp_path: Path, pilot_determined: dict[str, Any] | None = None) -> Path:
    data = yaml.safe_load(REAL_FAIR.read_text(encoding="utf-8"))
    data["pilot_determined"] = copy.deepcopy(SYNTHETIC_PILOT_DETERMINED if pilot_determined is None else pilot_determined)
    return _write_yaml(tmp_path / "fair_comparison.yaml", data)


def _all_off() -> dict[str, bool]:
    return {name: False for name in SWITCH_NAMES}


def _experiments(tmp_path: Path, **entries: dict[str, Any]) -> Path:
    return _write_yaml(tmp_path / "experiments.yaml", {"configs": entries})


def _valid_entry(**extra: Any) -> dict[str, Any]:
    return {"trainable": True, "switches": _all_off(), **extra}


# ---- 구성 로드와 덮어쓰기 거부 (US14-15) ----


@pytest.mark.parametrize("override_key", ["epochs", "learning_rate", "batch_size", "edge_loss_lambda", "seeds", "fixed_rules"])
def test_reject_fair_comparison_override(tmp_path: Path, override_key: str) -> None:
    """구성이 공정 비교 조건을 덮어쓰려 하면 실행 전에 거부되어야 합니다."""
    experiments = _experiments(tmp_path, evil=_valid_entry(**{override_key: 999}))
    with pytest.raises(ValueError, match="공정 비교 조건을 덮어쓰려"):
        load_experiment_config(experiments, "evil")
    with pytest.raises(ValueError, match="공정 비교 조건을 덮어쓰려"):
        resolve_run_config(_synthetic_fair(tmp_path), experiments, "evil", seed=0)


def test_reject_override_inside_switches(tmp_path: Path) -> None:
    """switches 안에 스위치가 아닌 키를 넣어 조건을 바꾸는 것도 거부해야 합니다."""
    experiments = _experiments(tmp_path, evil={"trainable": True, "switches": {**_all_off(), "epochs": 5}})
    with pytest.raises(ValueError, match="알 수 없는 키"):
        load_experiment_config(experiments, "evil")


@pytest.mark.parametrize(
    "entry",
    [
        {"trainable": True},  # switches 누락
        {"switches": _all_off()},  # trainable 누락
        {"trainable": True, "switches": {"A": False, "B": False, "C": False}},  # Bp 누락
        {"trainable": True, "switches": {**_all_off(), "A": "false"}},  # bool 아님
        {"trainable": "yes", "switches": _all_off()},
        {"trainable": True, "switches": _all_off(), "note": "memo"},  # 허용되지 않은 키
        {"trainable": True, "switches": _all_off(), "promoted_only": 1},
        "not-a-mapping",
    ],
)
def test_reject_malformed_experiment_entry(tmp_path: Path, entry: Any) -> None:
    experiments = _experiments(tmp_path, bad=entry)
    with pytest.raises(ValueError):
        load_experiment_config(experiments, "bad")


def test_reject_unknown_config_id(tmp_path: Path) -> None:
    experiments = _experiments(tmp_path, baseline=_valid_entry())
    with pytest.raises(ValueError, match="baseline"):
        load_experiment_config(experiments, "nope")


def test_experiments_file_top_level_must_be_configs_only(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "experiments.yaml", {"configs": {"baseline": _valid_entry()}, "epochs": 5})
    with pytest.raises(ValueError, match="configs"):
        load_experiment_config(path, "baseline")


def test_real_experiments_load_and_baseline_has_all_switches_off() -> None:
    """실제 configs/experiments.yaml의 10개 구성이 모두 로드되고, baseline은 스위치가 전부 꺼져 있다(US29)."""
    config_ids = yaml.safe_load(REAL_EXPERIMENTS.read_text(encoding="utf-8"))["configs"].keys()
    assert len(config_ids) == 10
    for config_id in config_ids:
        trainable, switches = load_experiment_config(REAL_EXPERIMENTS, config_id)
        assert isinstance(switches, Switches)
        assert trainable == (config_id != "bicubic")
    assert load_experiment_config(REAL_EXPERIMENTS, "baseline") == (True, Switches())
    assert load_experiment_config(REAL_EXPERIMENTS, "AB")[1] == Switches(A=True, B=True)
    assert load_experiment_config(REAL_EXPERIMENTS, "Bp")[1] == Switches(Bp=True)


# ---- 공정 비교 조건 파일 ----


def test_real_fair_comparison_loads_but_is_not_yet_determined() -> None:
    """T4.2 전에는 파일이 읽히지만 pilot_determined에 null이 남아 있어 실행 조건을 확정할 수 없다."""
    fair = load_fair_comparison(REAL_FAIR)
    assert fair["fixed_rules"]["seeds"] == [0, 1, 2]
    pending = find_undetermined(fair)
    if pending:  # T4.2 이후에는 비어 있어야 하므로 아래 확인은 고정 전에만 의미가 있다.
        assert "epochs" in pending
        assert "representative_images.Set5.median" in pending
        with pytest.raises(ValueError, match="null"):
            resolve_run_config(REAL_FAIR, REAL_EXPERIMENTS, "baseline", seed=0)
    else:
        assert resolve_run_config(REAL_FAIR, REAL_EXPERIMENTS, "baseline", seed=0).seed == 0


def test_find_undetermined_empty_for_fully_determined(tmp_path: Path) -> None:
    fair = load_fair_comparison(_synthetic_fair(tmp_path))
    assert find_undetermined(fair) == []


def test_fair_comparison_structure_errors(tmp_path: Path) -> None:
    base = yaml.safe_load(REAL_FAIR.read_text(encoding="utf-8"))

    def broken(mutate: Any) -> Path:
        data = copy.deepcopy(base)
        mutate(data)
        return _write_yaml(tmp_path / "broken.yaml", data)

    cases = {
        "missing section": lambda d: d.pop("infer_benchmark"),
        "extra section": lambda d: d.update(extra={}),
        "missing key": lambda d: d["fixed_rules"].pop("shave_border"),
        "typo key": lambda d: d["pilot_determined"].update(epoch=1),
        "null fixed rule": lambda d: d["fixed_rules"].update(scale=None),
        "bad seeds": lambda d: d["fixed_rules"].update(seeds=[0, "1"]),
        "bool seed": lambda d: d["fixed_rules"].update(seeds=[True]),
        "bad representative": lambda d: d["pilot_determined"]["representative_images"].update(Set5={"median": None}),
    }
    for name, mutate in cases.items():
        with pytest.raises(ValueError):
            load_fair_comparison(broken(mutate))
            pytest.fail(f"구조 오류가 거부되지 않았습니다: {name}")


def test_load_fair_comparison_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fair_comparison(tmp_path / "absent.yaml")


# ---- 확정 실행 조건 (US13, 28-29) ----


def test_resolved_configs_differ_only_by_switches_and_seed(tmp_path: Path) -> None:
    """서로 다른 구성의 확정 실행 조건은 스위치와 시드 외에 차이가 없어야 합니다."""
    fair = _synthetic_fair(tmp_path)
    config_ids = list(yaml.safe_load(REAL_EXPERIMENTS.read_text(encoding="utf-8"))["configs"])
    resolved = [
        resolve_run_config(fair, REAL_EXPERIMENTS, config_id, seed) for config_id in config_ids for seed in (0, 1, 2)
    ]
    assert len(resolved) == 30

    reference = resolved[0]
    allowed_to_differ = {"config_id", "seed", "trainable", "switches"}
    for other in resolved[1:]:
        differing = {
            field.name
            for field in dataclasses.fields(reference)
            if getattr(reference, field.name) != getattr(other, field.name)
        }
        assert differing <= allowed_to_differ
        assert other.fair_comparison == reference.fair_comparison
    # 같은 구성이라도 시드가 다르면 시드만 다르다.
    seed0 = resolve_run_config(fair, REAL_EXPERIMENTS, "AB", seed=0)
    seed1 = resolve_run_config(fair, REAL_EXPERIMENTS, "AB", seed=1)
    assert dataclasses.replace(seed0, seed=1) == seed1


def test_resolved_config_carries_switches_and_seed(tmp_path: Path) -> None:
    fair = _synthetic_fair(tmp_path)
    run = resolve_run_config(fair, REAL_EXPERIMENTS, "BC", seed=2)
    assert (run.config_id, run.seed, run.trainable) == ("BC", 2, True)
    assert run.switches == Switches(B=True, C=True)
    assert run.fair_comparison["pilot_determined"]["epochs"] == 1
    assert resolve_run_config(fair, REAL_EXPERIMENTS, "bicubic", seed=0).trainable is False


@pytest.mark.parametrize("seed", [3, -1, True])
def test_reject_seed_outside_fixed_seeds(tmp_path: Path, seed: Any) -> None:
    with pytest.raises(ValueError, match="시드"):
        resolve_run_config(_synthetic_fair(tmp_path), REAL_EXPERIMENTS, "baseline", seed=seed)


def test_reject_resolving_with_undetermined_values(tmp_path: Path) -> None:
    undetermined = copy.deepcopy(SYNTHETIC_PILOT_DETERMINED)
    undetermined["learning_rate"] = None
    undetermined["representative_images"]["Set5"]["worst"] = None
    fair = _synthetic_fair(tmp_path, undetermined)
    assert find_undetermined(load_fair_comparison(fair)) == ["learning_rate", "representative_images.Set5.worst"]
    with pytest.raises(ValueError, match="learning_rate"):
        resolve_run_config(fair, REAL_EXPERIMENTS, "baseline", seed=0)


def test_resolved_configs_do_not_share_mutable_state(tmp_path: Path) -> None:
    """한 확정 조건의 공정 비교 값을 고쳐도 다른 확정 조건에는 번지지 않아야 합니다."""
    fair = _synthetic_fair(tmp_path)
    first = resolve_run_config(fair, REAL_EXPERIMENTS, "baseline", seed=0)
    first.fair_comparison["pilot_determined"]["epochs"] = 999
    second = resolve_run_config(fair, REAL_EXPERIMENTS, "A", seed=0)
    assert second.fair_comparison["pilot_determined"]["epochs"] == 1


@pytest.mark.skip(reason="Skeleton test: implement alongside data.sampler")
def test_sampler_same_order_for_same_seed_across_configs() -> None:
    """같은 시드면 구성이 달라도 샘플러가 동일한 패치 순서를 생성해야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside data.sampler")
def test_sampler_different_order_for_different_seeds() -> None:
    """시드가 다르면 다른 패치 순서를 생성해야 합니다."""
