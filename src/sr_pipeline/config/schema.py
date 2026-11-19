"""공정 비교 조건 및 구성 스위치 해석·검증 (SPEC.md / ADR-0001).

구성은 허용된 개선 스위치(A, B, C, Bp)와 시드 외의 키를 가질 수 없으며,
공정 비교 조건을 덮어쓰려 하면 실행 시작 전에 거부합니다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SWITCH_NAMES: tuple[str, ...] = ("A", "B", "C", "Bp")
"""구성이 가질 수 있는 개선 스위치 이름 전부(통제 구성 B′는 `Bp`)."""

_EXPERIMENT_REQUIRED_KEYS = frozenset({"trainable", "switches"})
_EXPERIMENT_OPTIONAL_KEYS = frozenset({"promoted_only"})

_FIXED_RULES_KEYS = frozenset(
    {
        "scale",
        "degradation",
        "bicubic_rule",
        "color_standard",
        "eval_checkpoint",
        "max_hours",
        "seeds",
        "datasets",
        "shave_border",
        "lpips_backbone",
    }
)
_INFER_BENCHMARK_KEYS = frozenset(
    {
        "lr_height",
        "lr_width",
        "sr_height",
        "sr_width",
        "batch_size",
        "precision",
        "warmup_runs",
        "repeat_runs",
    }
)
_PILOT_DETERMINED_KEYS = frozenset(
    {
        "patch_pool_id",
        "patch_pool_size",
        "patch_size_hr",
        "patches_per_epoch",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "scheduler",
        "augmentation",
        "edge_loss_lambda",
        "edge_top_q",
        "representative_images",
    }
)
_REPRESENTATIVE_KEYS = frozenset({"median", "worst"})
_FAIR_SECTIONS: dict[str, frozenset[str]] = {
    "fixed_rules": _FIXED_RULES_KEYS,
    "infer_benchmark": _INFER_BENCHMARK_KEYS,
    "pilot_determined": _PILOT_DETERMINED_KEYS,
}
# 구성이 이 이름 중 하나를 가지면 "공정 비교 조건 덮어쓰기"로 따로 알린다.
_FAIR_COMPARISON_NAMES = frozenset(_FAIR_SECTIONS) | _FIXED_RULES_KEYS | _INFER_BENCHMARK_KEYS | _PILOT_DETERMINED_KEYS


@dataclass(frozen=True)
class Switches:
    """개선 스위치 조합."""

    A: bool = False
    B: bool = False
    C: bool = False
    Bp: bool = False


@dataclass(frozen=True)
class ResolvedRunConfig:
    """공정 비교 조건 + 구성 스위치 + 시드가 결합된 단일 실행 조건."""

    config_id: str
    seed: int
    trainable: bool
    switches: Switches
    fair_comparison: dict[str, Any]


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{label}의 최상위는 매핑이어야 합니다: {path}")
    return data


def _check_keys(mapping: dict[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - mapping.keys())
    if missing:
        raise ValueError(f"{where}에 필요한 키가 없습니다: {missing}")
    unknown = sorted(str(key) for key in mapping.keys() - expected)
    if unknown:
        raise ValueError(f"{where}에 알 수 없는 키가 있습니다: {unknown}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_fair_comparison(data: dict[str, Any], path: Path) -> None:
    _check_keys(data, frozenset(_FAIR_SECTIONS), f"{path.name}의 최상위")
    for section, keys in _FAIR_SECTIONS.items():
        values = data[section]
        if not isinstance(values, dict):
            raise ValueError(f"{path.name}의 {section}는 매핑이어야 합니다: {values!r}")
        _check_keys(values, keys, f"{path.name}의 {section}")

    for section in ("fixed_rules", "infer_benchmark"):
        for key, value in data[section].items():
            if value is None:
                raise ValueError(f"{path.name}의 {section}.{key}는 null일 수 없습니다(고정 값).")

    seeds = data["fixed_rules"]["seeds"]
    if not isinstance(seeds, list) or not seeds or not all(_is_int(seed) for seed in seeds):
        raise ValueError(f"fixed_rules.seeds는 비어 있지 않은 정수 목록이어야 합니다: {seeds!r}")
    datasets = data["fixed_rules"]["datasets"]
    if not isinstance(datasets, list) or not datasets or not all(isinstance(name, str) and name for name in datasets):
        raise ValueError(f"fixed_rules.datasets는 비어 있지 않은 문자열 목록이어야 합니다: {datasets!r}")

    representative = data["pilot_determined"]["representative_images"]
    if not isinstance(representative, dict):
        raise ValueError(f"pilot_determined.representative_images는 매핑이어야 합니다: {representative!r}")
    for dataset, entry in representative.items():
        if not isinstance(entry, dict):
            raise ValueError(f"pilot_determined.representative_images.{dataset}는 매핑이어야 합니다: {entry!r}")
        _check_keys(entry, _REPRESENTATIVE_KEYS, f"pilot_determined.representative_images.{dataset}")


def load_fair_comparison(path: Path) -> dict[str, Any]:
    """`configs/fair_comparison.yaml`을 로드하고 구조를 검증합니다.

    `pilot_determined`의 `null`(T4.2 확정 전)은 허용합니다. 확정 여부는 `find_undetermined`와
    `resolve_run_config`가 따로 확인합니다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: 최상위·절 구조, 필수·알 수 없는 키, 고정 값의 `null`이 잘못되었을 때.
    """
    path = Path(path)
    data = _read_yaml_mapping(path, "공정 비교 조건")
    _validate_fair_comparison(data, path)
    return data


def find_undetermined(fair_comparison: dict[str, Any]) -> list[str]:
    """`pilot_determined`에서 아직 `null`인 항목의 점 경로 목록을 돌려줍니다(T4.2 완료면 빈 목록)."""
    pending: list[str] = []

    def walk(value: Any, prefix: str) -> None:
        if value is None:
            pending.append(prefix)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}")

    for key, value in fair_comparison["pilot_determined"].items():
        walk(value, key)
    return pending


def _parse_switches(raw: Any, where: str) -> Switches:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}의 switches는 매핑이어야 합니다: {raw!r}")
    _check_keys(raw, frozenset(SWITCH_NAMES), f"{where}의 switches")
    for name in SWITCH_NAMES:
        if not isinstance(raw[name], bool):
            raise ValueError(f"{where}의 switches.{name}는 true/false여야 합니다: {raw[name]!r}")
    return Switches(**{name: raw[name] for name in SWITCH_NAMES})


def load_experiment_config(path: Path, config_id: str) -> tuple[bool, Switches]:
    """`configs/experiments.yaml`에서 구성 정의를 로드하고 허용되지 않은 키를 거부합니다.

    구성은 `trainable`, `promoted_only`(선택), `switches`(A·B·C·Bp)만 가질 수 있습니다.
    공정 비교 조건 이름(`epochs`, `learning_rate` 등)을 가지면 덮어쓰기 시도로 거부합니다.

    Raises:
        ValueError: 구성 정의가 잘못되었거나 `config_id`가 없을 때.
    """
    path = Path(path)
    data = _read_yaml_mapping(path, "실험 구성")
    if set(data) != {"configs"} or not isinstance(data["configs"], dict):
        raise ValueError(f"{path.name}의 최상위는 configs 매핑 하나여야 합니다.")
    configs = data["configs"]
    if config_id not in configs:
        raise ValueError(f"알 수 없는 구성 {config_id!r}입니다. 가능한 구성: {sorted(configs)}")

    entry = configs[config_id]
    where = f"구성 {config_id}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}는 매핑이어야 합니다: {entry!r}")

    overrides = sorted(str(key) for key in entry if key in _FAIR_COMPARISON_NAMES)
    if overrides:
        raise ValueError(
            f"{where}가 공정 비교 조건을 덮어쓰려 합니다: {overrides}. "
            "구성은 trainable과 switches만 정의할 수 있고, 공정 비교 조건은 fair_comparison.yaml에서만 바꿉니다."
        )
    allowed = _EXPERIMENT_REQUIRED_KEYS | _EXPERIMENT_OPTIONAL_KEYS
    unknown = sorted(str(key) for key in entry if key not in allowed)
    if unknown:
        raise ValueError(f"{where}에 허용되지 않은 키가 있습니다: {unknown}. 허용: {sorted(allowed)}")
    missing = sorted(_EXPERIMENT_REQUIRED_KEYS - entry.keys())
    if missing:
        raise ValueError(f"{where}에 필요한 키가 없습니다: {missing}")

    for key in ("trainable", "promoted_only"):
        if key in entry and not isinstance(entry[key], bool):
            raise ValueError(f"{where}의 {key}는 true/false여야 합니다: {entry[key]!r}")
    return entry["trainable"], _parse_switches(entry["switches"], where)


def resolve_run_config(
    fair_path: Path,
    experiments_path: Path,
    config_id: str,
    seed: int,
) -> ResolvedRunConfig:
    """단일 실행에 사용할 최종 실행 조건을 확정합니다.

    공정 비교 조건이 모든 구성에 같은 값으로 들어가므로, 확정된 실행 조건끼리의 차이는
    `config_id`·`trainable`·`switches`·`seed`뿐입니다.

    Raises:
        ValueError: 시드가 `fixed_rules.seeds`에 없거나, `pilot_determined`에 `null`이 남아 있거나(T4.2 전),
            구성·공정 비교 조건 파일이 잘못되었을 때.
    """
    fair_comparison = load_fair_comparison(fair_path)
    trainable, switches = load_experiment_config(experiments_path, config_id)

    allowed_seeds = fair_comparison["fixed_rules"]["seeds"]
    if not _is_int(seed) or seed not in allowed_seeds:
        raise ValueError(f"시드 {seed!r}는 fixed_rules.seeds {allowed_seeds}에 없습니다.")
    pending = find_undetermined(fair_comparison)
    if pending:
        raise ValueError(
            "pilot_determined에 아직 null인 값이 있어 실행 조건을 확정할 수 없습니다(T4.2에서 고정): "
            f"{pending}"
        )
    return ResolvedRunConfig(
        config_id=config_id,
        seed=seed,
        trainable=trainable,
        switches=switches,
        fair_comparison=copy.deepcopy(fair_comparison),
    )
