"""실행 기록 자동 집계 및 보고서/발표용 표·그림 생성 (SPEC.md / ADR-0003).

원칙:
- main에 병합된 실행 기록(`results/runs/`)만 입력으로 사용
- `(config_id, seed, patch_pool_id)` 중복 감지 시 오류
- 시드 3개 미만 또는 `incomplete` 포함 구성은 "미완료" 표시
- 평균 ± 표준편차, 베이스라인 대비 Δ, 시드별 Δ 부호 일관성, 부호 있는 평균 Δ > std(seed Δ) 판정
- Ablation 표(A: RGB·Y 맥락, B: 정렬 B′·입력/용량 B 분해), 영상별 Δ 분포, 사례 영상 자동 선정

이 파일의 구현 범위(T4.3 집계 핵심): `load_run_records`, `validate_unique_runs`, `compute_result_tables`.
Ablation 표 조립·영상별 Δ·사례 영상·그림·`export_report_assets`는 T6.3에서 구현한다.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from sr_pipeline.config.schema import load_fair_comparison

SUMMARY_FILENAME = "summary.csv"
PER_IMAGE_FILENAME = "per_image.csv"

BASELINE_CONFIG_ID = "baseline"
BICUBIC_CONFIG_ID = "bicubic"  # 학습 없는 하한. 시드 3개 요건과 시드별 Δ에서 제외한다.

METRICS = ("psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips")
_STATUSES = ("complete", "incomplete")
_RUN_KEY = ["config_id", "seed", "patch_pool_id"]
_GROUP_KEY = ["patch_pool_id", "dataset", "config_id"]

_SUMMARY_COLUMNS = (
    "run_id", "commit", "config_id", "seed", "patch_pool_id", "switches", "dataset",
    *METRICS, "psnr_y_pre_rgb", "params", "infer_ms_net", "infer_ms_total", "train_minutes", "status",
)  # fmt: skip
_PER_IMAGE_COLUMNS = ("run_id", "dataset", "image_id", *METRICS)
_STRING_COLUMNS = ("run_id", "commit", "config_id", "patch_pool_id", "switches", "dataset", "image_id", "status")
# SPEC "실행 기록 형식"에서 빈 값이 허용되는 열은 psnr_y_pre_rgb(B 계열 진단값) 하나뿐이다.
_SUMMARY_STRINGS = ("run_id", "commit", "config_id", "patch_pool_id", "switches", "dataset", "status")
_SUMMARY_NUMBERS = ("seed", *METRICS, "params", "infer_ms_net", "infer_ms_total", "train_minutes")
_SUMMARY_OPTIONAL_NUMBERS = ("psnr_y_pre_rgb",)
_PSNR_COLUMNS = frozenset({"psnr_y", "psnr_rgb", "edge_psnr", "psnr_y_pre_rgb"})
_SUMMARY_INTEGERS = ("seed", "params")
_PER_IMAGE_STRINGS = ("run_id", "dataset", "image_id")
_KEY_COLUMNS = ("run_id", *_RUN_KEY, "dataset")

_RESULT_COLUMNS = [
    "patch_pool_id", "dataset", "config_id", "metric",
    "n_seeds", "seeds", "mean", "std", "delta",
    "seed_delta_std", "delta_sign", "delta_sign_consistent", "delta_exceeds_seed_delta_std",
    "is_complete", "missing_seeds", "incomplete_seeds", "baseline_is_complete",
    "run_ids", "incomplete_run_ids", "baseline_run_ids",
]  # fmt: skip
_SEED_DELTA_COLUMNS = [*_GROUP_KEY, "metric", "seed", "delta", "run_id", "baseline_run_id"]
_PER_IMAGE_SEED_DELTA_COLUMNS = [
    "patch_pool_id", "dataset", "config_id", "image_id", "metric", "seed",
    "value", "baseline_value", "delta", "run_id", "baseline_run_id",
]  # fmt: skip
_PER_IMAGE_SUMMARY_COLUMNS = [
    "patch_pool_id", "dataset", "config_id", "image_id", "metric",
    "n_seeds", "seeds", "mean_delta", "std_delta", "delta_sign", "run_ids", "baseline_run_ids",
]  # fmt: skip
LOWER_IS_BETTER_METRICS = frozenset({"lpips"})


def _read_csv(path: Path) -> pd.DataFrame:
    # "none"(스위치 없음) 같은 문자열이 결측으로 바뀌지 않게 하고, 빈 칸만 결측으로 읽는다.
    # 식별자 열은 숫자처럼 보여도(예: 숫자만 있는 짧은 커밋) 문자열로 읽는다.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={c: str for c in _STRING_COLUMNS})
    if df.empty:
        raise ValueError(f"{path}에 행이 없습니다.")
    return df


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _check_no_blanks(df: pd.DataFrame, columns: Sequence[str], source: object) -> None:
    """빈 값·공백뿐인 키는 groupby가 조용히 버리거나 엉뚱한 그룹을 만들므로 집계 전에 거부한다."""
    blank = [c for c in columns if _is_blank(df[c]).any()]
    if blank:
        raise ValueError(f"{source}에 빈 값이 있는 필수 열이 있습니다: {blank}")


def _check_statuses(df: pd.DataFrame, source: object) -> None:
    _check_no_blanks(df, ("status",), source)
    bad = sorted(set(df["status"]) - set(_STATUSES))
    if bad:
        raise ValueError(f"{source}의 status는 {_STATUSES} 중 하나여야 합니다: {bad}")


def _as_valid_numbers(
    df: pd.DataFrame,
    required: Sequence[str],
    source: object,
    *,
    optional: Sequence[str] = (),
    allow_posinf: Iterable[str] = (),
) -> pd.DataFrame:
    """수치 열을 float로 바꾼 사본을 돌려준다.

    `optional` 열만 빈 값을 허용한다(빈 값은 NaN으로 남긴다).
    `allow_posinf`에 포함된 PSNR 계열 열은 오차 0에서 평가기가 기록하는 양의 무한대(`+inf`)를 허용하고,
    그 밖의 `NaN`·`-inf`·`+inf`·숫자가 아닌 값은 거부한다.
    """
    df = df.copy()
    posinf_allowed = frozenset(allow_posinf)
    bad: list[str] = []
    for column in (*required, *optional):
        blank = _is_blank(df[column])
        values = pd.to_numeric(df[column].where(~blank), errors="coerce").astype(float)
        valid_num = np.isfinite(values)
        if column in posinf_allowed:
            valid_num = valid_num | np.isposinf(values)
        invalid = ~blank & ~valid_num
        if column in required:
            invalid |= blank
        if invalid.any():
            bad.append(column)
        df[column] = values
    if bad:
        raise ValueError(f"{source}에 빈 값이거나 허용되지 않은 숫자가 있는 열이 있습니다: {bad}")
    return df


def _check_columns(df: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}에 필요한 열이 없습니다: {missing}")


def load_run_records(
    runs_dir: Path,
    *,
    allowed_run_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`runs_dir/<run_id>/{summary,per_image}.csv`를 모두 읽어 (summary, per_image)로 합칩니다.

    실행 폴더에 두 파일 중 하나라도 없거나, 파일에 행이 없거나, 필요한 열이 없거나,
    필수 값이 비었거나(빈 값은 `psnr_y_pre_rgb`만 허용), 허용되지 않은 수치(`+inf`는 PSNR 계열 열만 허용,
    `NaN`·`-inf`·비숫자는 모두 거부)가 있거나, `status`가 알 수 없는 값이면 집계 전에 `ValueError`로 멈춘다.
    `allowed_run_ids`를 주면 그 목록(예: `main`에 병합된 실행)에 속한 폴더만 읽는다.
    중복 검사는 `validate_unique_runs`가 한다.
    "main에 병합된 기록만"이라는 조건은 T6.3 집계 스크립트(`scripts/aggregate_results.py`)에서 검증한다.
    """
    runs_dir = Path(runs_dir)
    allowed_set = None if allowed_run_ids is None else {str(r) for r in allowed_run_ids}
    run_dirs = (
        sorted(
            d
            for d in {
                p.parent
                for name in (SUMMARY_FILENAME, PER_IMAGE_FILENAME)
                for p in runs_dir.glob(f"*/{name}")
            }
            if allowed_set is None or d.name in allowed_set
        )
        if runs_dir.is_dir()
        else []
    )
    if not run_dirs:
        raise ValueError(f"실행 기록이 없습니다(<실행 폴더>/{SUMMARY_FILENAME}를 찾지 못함): {runs_dir}")

    summaries: list[pd.DataFrame] = []
    per_images: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        summary_path = run_dir / SUMMARY_FILENAME
        per_image_path = run_dir / PER_IMAGE_FILENAME
        for path in (summary_path, per_image_path):
            if not path.is_file():
                raise ValueError(f"{run_dir}에 {path.name}가 없습니다.")
        summary = _read_csv(summary_path)
        per_image = _read_csv(per_image_path)
        _check_columns(summary, _SUMMARY_COLUMNS, summary_path)
        _check_columns(per_image, _PER_IMAGE_COLUMNS, per_image_path)
        _check_no_blanks(summary, _SUMMARY_STRINGS, summary_path)
        _check_no_blanks(per_image, _PER_IMAGE_STRINGS, per_image_path)
        summary = _as_valid_numbers(
            summary,
            _SUMMARY_NUMBERS,
            summary_path,
            optional=_SUMMARY_OPTIONAL_NUMBERS,
            allow_posinf=_PSNR_COLUMNS,
        )
        fractional = [c for c in _SUMMARY_INTEGERS if (summary[c] % 1 != 0).any()]
        if fractional:
            raise ValueError(f"{summary_path}의 정수 열에 소수가 있습니다: {fractional}")
        summary = summary.astype({c: int for c in _SUMMARY_INTEGERS})
        per_image = _as_valid_numbers(per_image, METRICS, per_image_path, allow_posinf=_PSNR_COLUMNS)
        summaries.append(summary[list(_SUMMARY_COLUMNS)])
        per_images.append(per_image[list(_PER_IMAGE_COLUMNS)])

    summary_df = pd.concat(summaries, ignore_index=True)
    per_image_df = pd.concat(per_images, ignore_index=True)

    _check_statuses(summary_df, "summary")
    orphan = sorted(set(per_image_df["run_id"]) - set(summary_df["run_id"]))
    if orphan:
        raise ValueError(f"summary에 없는 실행의 per_image 기록이 있습니다: {orphan}")
    return summary_df, per_image_df


def validate_unique_runs(summary_df: pd.DataFrame) -> None:
    """`(config_id, seed, patch_pool_id)` 조합이 둘 이상의 실행에 나타나면 `ValueError`를 냅니다.

    실행 1회는 테스트셋마다 summary 행을 하나씩 만들므로, 한 실행 안에서 같은 테스트셋 행이
    두 번 나오는 것도 중복이다. 커밋이 달라 `run_id`가 다른 재실행도 같은 조합이면 중복이다.
    키 열(`run_id`, `config_id`, `seed`, `patch_pool_id`, `dataset`)이 비어 있으면 groupby가 그 행을
    조용히 버려 중복을 놓치므로, DataFrame을 직접 넘긴 경우에도 먼저 `ValueError`로 거부한다.
    """
    _check_no_blanks(summary_df, _KEY_COLUMNS, "summary_df")
    problems: list[str] = []
    flagged: set[tuple] = set()
    run_ids_per_key = summary_df.groupby(_RUN_KEY)["run_id"].agg(lambda s: sorted(s.unique()))
    for key, run_ids in run_ids_per_key.items():
        if len(run_ids) > 1:
            flagged.add(key)
            problems.append(f"{dict(zip(_RUN_KEY, key))}: 실행 {run_ids}")
    rows_per_dataset = summary_df.groupby([*_RUN_KEY, "dataset"]).size()
    for key, n_rows in rows_per_dataset.items():
        if n_rows > 1 and key[:3] not in flagged:
            problems.append(f"{dict(zip([*_RUN_KEY, 'dataset'], key))}: summary 행 {n_rows}개")
    if problems:
        raise ValueError("(config_id, seed, patch_pool_id) 중복 실행 기록이 있습니다:\n  " + "\n  ".join(problems))


def _sample_std(values: Sequence[float]) -> float:
    """표본 표준편차(ddof=1). 값이 2개 미만이거나 유한하지 않은 값(inf·NaN)이 있으면 정의되지 않아 NaN."""
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        return math.nan
    return float(np.std(values, ddof=1))


def _join_sorted(items: Iterable[Any]) -> str:
    return ",".join(str(item) for item in sorted(items))


def _delta_sign(seed_deltas: Sequence[float]) -> str:
    """모두 양수 `positive`, 모두 음수 `negative`, 그 외(0 포함) `mixed`, 짝이 없거나 NaN이 있으면 빈 문자열."""
    if not seed_deltas or any(math.isnan(d) for d in seed_deltas):
        return ""
    if all(d > 0 for d in seed_deltas):
        return "positive"
    if all(d < 0 for d in seed_deltas):
        return "negative"
    return "mixed"


def _validate_summary_input(summary_df: pd.DataFrame, expected_seeds: Sequence[int]) -> tuple[list[int], pd.DataFrame]:
    expected = sorted(set(int(s) for s in expected_seeds))
    if not expected:
        raise ValueError("expected_seeds가 비어 있습니다.")
    validate_unique_runs(summary_df)
    _check_statuses(summary_df, "summary_df")
    validated = _as_valid_numbers(summary_df, METRICS, "summary_df", allow_posinf=_PSNR_COLUMNS)
    unexpected = sorted(set(int(s) for s in validated["seed"]) - set(expected))
    if unexpected:
        raise ValueError(f"fair_comparison의 시드 {expected}에 없는 시드가 있습니다: {unexpected}")
    return expected, validated


def _paired_contrast_stats(
    target_by_seed: pd.DataFrame,
    ref_by_seed: pd.DataFrame,
    expected: Sequence[int],
    metric: str,
    *,
    pair_seeds: bool = True,
) -> dict[str, Any]:
    """완료된 target·reference 시드 행으로 평균 차이 Δ, 시드별 Δ, 부호 일관성, 평균 Δ > std 판정을 구한다."""
    has_target = len(target_by_seed) > 0
    has_ref = len(ref_by_seed) > 0
    has_delta = has_target and has_ref
    target_mean = float(np.mean(list(target_by_seed[metric]))) if has_target else math.nan
    ref_mean = float(np.mean(list(ref_by_seed[metric]))) if has_ref else math.nan
    delta = (target_mean - ref_mean) if has_delta else math.nan
    paired = sorted(set(target_by_seed.index) & set(ref_by_seed.index)) if (pair_seeds and has_delta) else []
    seed_deltas = [float(target_by_seed.at[s, metric]) - float(ref_by_seed.at[s, metric]) for s in paired]
    sign = _delta_sign(seed_deltas)
    seed_delta_std = _sample_std(seed_deltas)
    consistent: Any = pd.NA
    exceeds: Any = pd.NA
    if paired == list(expected) and sign != "":
        consistent = sign != "mixed"
        if not math.isnan(seed_delta_std):
            mean_seed_delta = float(np.mean(seed_deltas))
            if math.isfinite(mean_seed_delta):
                exceeds = mean_seed_delta > seed_delta_std
    return {
        "target_mean": target_mean,
        "ref_mean": ref_mean,
        "delta": delta,
        "paired": paired,
        "seed_deltas": seed_deltas,
        "seed_delta_std": seed_delta_std,
        "delta_sign": sign,
        "delta_sign_consistent": consistent,
        "delta_exceeds_seed_delta_std": exceeds,
    }


def compute_result_tables(summary_df: pd.DataFrame, *, expected_seeds: Sequence[int]) -> dict[str, pd.DataFrame]:
    """평균 ± 표준편차, Δ, 시드별 Δ 일관성, 미완료 표시를 계산합니다.

    `expected_seeds`는 `fair_comparison.yaml`의 `fixed_rules.seeds`를 호출자가 넘긴다(기본값 없음).
    (패치 풀, 테스트셋, 구성, 지표)마다 한 행인 `results`와, 시드별 Δ를 모두 담은 `seed_deltas`를 돌려준다.

    - 완료 실행만 수치에 포함(SPEC US20): `status=incomplete`(3시간 상한 중단 등) 실행은 `mean`·`std`·`delta`·
      시드 짝짓기에서 제외하고, `is_complete=False`와 `incomplete_seeds`·`incomplete_run_ids`에만 남긴다.
    - `std`·`seed_delta_std`: 표본 표준편차(ddof=1), 값이 2개 미만이거나 `+inf` 등으로 정의되지 않으면 NaN.
    - `delta`: 같은 패치 풀·테스트셋·지표에서 (구성의 완료 시드 평균) − (베이스라인의 완료 시드 평균).
      베이스라인 자신, 완료 실행이 없거나 완료 베이스라인이 없는 경우는 NaN. `inf - inf`도 NaN.
    - `delta_sign`: 같은 완료 시드끼리 짝지은 시드별 Δ가 모두 양수면 `positive`, 모두 음수면 `negative`,
      그 외(0 포함)는 `mixed`. 짝이 없거나 `inf - inf` 등 NaN이 섞이면 빈 문자열(판정 불가).
      베이스라인·Bicubic 하한·짝지을 완료 시드가 없으면 빈 문자열.
    - `delta_sign_consistent`, `delta_exceeds_seed_delta_std`: SPEC US59·ADR-0003의 "시드별 Δ 부호가 3개 모두
      같은가", "평균 Δ가 시드별 Δ의 표준편차보다 큰가" 판정. 기대 시드가 **모두** 완료 실행으로 베이스라인과
      짝지어졌을 때만 값을 내고, 하나라도 빠지거나 Δ·표준편차가 정의되지 않으면 `<NA>`다.
      두 번째 판정은 SPEC·ADR 문구대로 부호 있는 평균 Δ(`mean(seed_deltas) > seed_delta_std`)를 비교한다
      (낮을수록 좋은 LPIPS의 해석은 T6.3 보고서 표에서 따로 다룬다).
    - `is_complete`: 기대 시드가 모두 있고 모두 `status=complete`. Bicubic 하한은 시드 하나면 충분하다.
    - `baseline_is_complete`: 같은 패치 풀·테스트셋의 베이스라인이 완료인지(Δ의 기준이 미완료인지 알리기 위함).
    - `run_ids`·`baseline_run_ids`(`seed_deltas`에서는 `run_id`·`baseline_run_id`): 그 수치를 만든 완료 실행 ID
      (ADR-0006 추적성). `incomplete_run_ids`는 미완료 실행 ID(진행 현황용). Δ가 없는 행의 `baseline_run_ids`는 빈 문자열이다.

    `load_run_records`를 거치지 않은 DataFrame도 받으므로, 키 열 결측·`status`·허용되지 않은 지표 값은 여기서도 거부한다.
    """
    expected, summary_df = _validate_summary_input(summary_df, expected_seeds)

    result_rows: list[dict[str, Any]] = []
    seed_delta_rows: list[dict[str, Any]] = []

    for (pool, dataset), block in summary_df.groupby(["patch_pool_id", "dataset"]):
        baseline_all = block[block["config_id"] == BASELINE_CONFIG_ID]
        baseline_complete_rows = baseline_all[baseline_all["status"] == "complete"].set_index("seed")
        baseline_complete = None
        if len(baseline_all):
            baseline_complete, _, _ = _is_complete(baseline_all, expected, trainable=True)

        for config_id, config_rows in block.groupby("config_id"):
            trainable = config_id != BICUBIC_CONFIG_ID
            complete, missing, incomplete = _is_complete(config_rows, expected, trainable=trainable)
            complete_rows = config_rows[config_rows["status"] == "complete"]
            incomplete_rows = config_rows[config_rows["status"] != "complete"]
            by_seed = complete_rows.set_index("seed")
            has_delta = config_id != BASELINE_CONFIG_ID and len(baseline_complete_rows) > 0 and len(by_seed) > 0

            for metric in METRICS:
                values = list(by_seed[metric])
                if config_id == BASELINE_CONFIG_ID:
                    mean = float(np.mean(values)) if values else math.nan
                    stats = {
                        "delta": math.nan, "paired": [], "seed_deltas": [],
                        "seed_delta_std": math.nan, "delta_sign": "",
                        "delta_sign_consistent": pd.NA, "delta_exceeds_seed_delta_std": pd.NA,
                    }  # fmt: skip
                else:
                    stats = _paired_contrast_stats(
                        by_seed, baseline_complete_rows, expected, metric, pair_seeds=trainable
                    )
                    mean = stats["target_mean"]
                for s, d in zip(stats["paired"], stats["seed_deltas"]):
                    seed_delta_rows.append(
                        {"patch_pool_id": pool, "dataset": dataset, "config_id": config_id, "metric": metric,
                         "seed": int(s), "delta": d,
                         "run_id": by_seed.at[s, "run_id"], "baseline_run_id": baseline_complete_rows.at[s, "run_id"]}
                    )  # fmt: skip

                result_rows.append(
                    {
                        "patch_pool_id": pool, "dataset": dataset, "config_id": config_id, "metric": metric,
                        "n_seeds": len(values), "seeds": _join_sorted(by_seed.index),
                        "mean": mean, "std": _sample_std(values), "delta": stats["delta"],
                        "seed_delta_std": stats["seed_delta_std"], "delta_sign": stats["delta_sign"],
                        "delta_sign_consistent": stats["delta_sign_consistent"],
                        "delta_exceeds_seed_delta_std": stats["delta_exceeds_seed_delta_std"],
                        "is_complete": complete, "missing_seeds": _join_sorted(missing),
                        "incomplete_seeds": _join_sorted(incomplete),
                        "baseline_is_complete": baseline_complete if baseline_complete is not None else pd.NA,
                        "run_ids": _join_sorted(by_seed["run_id"]),
                        "incomplete_run_ids": _join_sorted(incomplete_rows["run_id"]),
                        "baseline_run_ids": _join_sorted(baseline_complete_rows["run_id"]) if has_delta else "",
                    }
                )  # fmt: skip

    results = pd.DataFrame(result_rows, columns=_RESULT_COLUMNS)
    for column in ("delta_sign_consistent", "delta_exceeds_seed_delta_std", "baseline_is_complete"):
        results[column] = results[column].astype("boolean")
    seed_deltas_df = pd.DataFrame(seed_delta_rows, columns=_SEED_DELTA_COLUMNS)
    return {"results": results, "seed_deltas": seed_deltas_df}


def _is_complete(rows: pd.DataFrame, expected: Sequence[int], *, trainable: bool) -> tuple[bool, list[int], list[int]]:
    """`(완료 여부, 빠진 시드, status가 incomplete인 시드)`. 학습 없는 구성은 행이 있고 모두 complete면 완료."""
    present = set(int(s) for s in rows["seed"])
    missing = sorted(set(expected) - present) if trainable else []
    incomplete = sorted(int(s) for s in rows.loc[rows["status"] != "complete", "seed"])
    return (not missing and not incomplete and len(rows) > 0), missing, incomplete


def _validate_per_image_input(summary_df: pd.DataFrame, per_image_df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(per_image_df, _PER_IMAGE_COLUMNS, Path("per_image_df"))
    _check_no_blanks(per_image_df, _PER_IMAGE_STRINGS, "per_image_df")
    validated = _as_valid_numbers(per_image_df, METRICS, "per_image_df", allow_posinf=_PSNR_COLUMNS)
    orphan = sorted(set(validated["run_id"]) - set(summary_df["run_id"]))
    if orphan:
        raise ValueError(f"summary에 없는 실행의 per_image 기록이 있습니다: {orphan}")
    duplicates = validated.groupby(["run_id", "dataset", "image_id"]).size()
    bad = duplicates[duplicates > 1]
    if not bad.empty:
        sample = [dict(zip(["run_id", "dataset", "image_id"], k)) for k in bad.index[:5]]
        raise ValueError(f"per_image_df에 중복 (run_id, dataset, image_id) 행이 있습니다: {sample}")
    return validated


def compute_per_image_deltas(
    summary_df: pd.DataFrame,
    per_image_df: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    baseline_config_id: str = BASELINE_CONFIG_ID,
) -> dict[str, pd.DataFrame]:
    """같은 테스트셋·영상·시드·패치 풀에서 완료된 구성과 베이스라인의 영상별 Δ를 계산합니다.

    `status=complete` 실행만 짝지어 `per_image_seed_deltas`(시드별 영상 Δ)와
    `per_image_summary`(영상별 시드 평균 Δ, 표본 표준편차, 부호 일관성, 실행 ID 추적)를 돌려줍니다.
    Bicubic 하한은 시드별 비교에서 제외합니다.
    """
    _, summary_df = _validate_summary_input(summary_df, expected_seeds)
    per_image_df = _validate_per_image_input(summary_df, per_image_df)

    run_meta = summary_df[["run_id", "patch_pool_id", "config_id", "seed", "dataset", "status"]].drop_duplicates()
    complete_runs = run_meta[run_meta["status"] == "complete"].drop(columns=["status"])
    joined = per_image_df.merge(complete_runs, on=["run_id", "dataset"], how="inner")

    base_rows = joined[joined["config_id"] == baseline_config_id]
    target_rows = joined[(joined["config_id"] != baseline_config_id) & (joined["config_id"] != BICUBIC_CONFIG_ID)]
    paired = target_rows.merge(
        base_rows,
        on=["patch_pool_id", "dataset", "image_id", "seed"],
        suffixes=("", "_baseline"),
        how="inner",
    )

    seed_rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        for metric in METRICS:
            val = float(getattr(row, metric))
            base_val = float(getattr(row, f"{metric}_baseline"))
            seed_rows.append(
                {
                    "patch_pool_id": row.patch_pool_id,
                    "dataset": row.dataset,
                    "config_id": row.config_id,
                    "image_id": row.image_id,
                    "metric": metric,
                    "seed": int(row.seed),
                    "value": val,
                    "baseline_value": base_val,
                    "delta": val - base_val,
                    "run_id": row.run_id,
                    "baseline_run_id": row.run_id_baseline,
                }
            )
    seed_df = pd.DataFrame(seed_rows, columns=_PER_IMAGE_SEED_DELTA_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    if not seed_df.empty:
        group_cols = ["patch_pool_id", "dataset", "config_id", "image_id", "metric"]
        for key, group in seed_df.groupby(group_cols, sort=True):
            ordered = group.sort_values("seed")
            deltas = [float(d) for d in ordered["delta"]]
            summary_rows.append(
                {
                    **dict(zip(group_cols, key)),
                    "n_seeds": len(ordered),
                    "seeds": _join_sorted(ordered["seed"]),
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": _sample_std(deltas),
                    "delta_sign": _delta_sign(deltas),
                    "run_ids": _join_sorted(ordered["run_id"]),
                    "baseline_run_ids": _join_sorted(ordered["baseline_run_id"]),
                }
            )
    summary_out = pd.DataFrame(summary_rows, columns=_PER_IMAGE_SUMMARY_COLUMNS)
    return {"per_image_summary": summary_out, "per_image_seed_deltas": seed_df}


def _to_per_image_summary_table(
    per_image_df: pd.DataFrame,
    config_id: str,
    *,
    summary_df: pd.DataFrame | None,
    expected_seeds: Sequence[int] | None,
    metric: str,
    baseline_config_id: str,
) -> pd.DataFrame:
    if summary_df is not None:
        seeds = list(expected_seeds) if expected_seeds is not None else sorted(set(int(s) for s in summary_df["seed"]))
        table = compute_per_image_deltas(
            summary_df, per_image_df, expected_seeds=seeds, baseline_config_id=baseline_config_id
        )["per_image_summary"]
        return table[(table["config_id"] == config_id) & (table["metric"] == metric)].copy()

    if "config_id" in per_image_df.columns and (
        "mean_delta" in per_image_df.columns or "delta" in per_image_df.columns
    ):
        filtered = per_image_df[per_image_df["config_id"] == config_id].copy()
        if "metric" in filtered.columns:
            filtered = filtered[filtered["metric"] == metric]
        if "patch_pool_id" not in filtered.columns:
            filtered["patch_pool_id"] = ""
        if "mean_delta" in filtered.columns:
            if "n_seeds" not in filtered.columns:
                filtered["n_seeds"] = 1
            for col in ("run_ids", "baseline_run_ids"):
                if col not in filtered.columns:
                    filtered[col] = ""
            return filtered
        grouped = (
            filtered.groupby(["patch_pool_id", "dataset", "config_id", "image_id"], as_index=False)
            .agg(mean_delta=("delta", "mean"), n_seeds=("delta", "size"))
        )
        grouped["metric"] = metric
        grouped["run_ids"] = ""
        grouped["baseline_run_ids"] = ""
        return grouped

    raise ValueError("사례 영상 선정에는 summary_df 또는 (config_id, mean_delta/delta) 열이 필요합니다.")


def _case_study_records(ranked_df: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    return [
        {
            "patch_pool_id": row.patch_pool_id,
            "dataset": row.dataset,
            "config_id": row.config_id,
            "image_id": row.image_id,
            "metric": metric,
            "mean_delta": float(row.mean_delta),
            "n_seeds": int(row.n_seeds),
            "run_ids": str(row.run_ids),
            "baseline_run_ids": str(row.baseline_run_ids),
            "rank": rank,
        }
        for rank, row in enumerate(ranked_df.itertuples(index=False), start=1)
    ]


def select_case_study_images(
    per_image_df: Any,
    config_id: str,
    *,
    summary_df: pd.DataFrame | None = None,
    expected_seeds: Sequence[int] | None = None,
    metric: str = "psnr_y",
    datasets: Sequence[str] | None = None,
    top_k: int = 1,
    baseline_config_id: str = BASELINE_CONFIG_ID,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """영상별 Δ의 상위·하위에서 성공·실패 사례 영상을 규칙으로 자동 선정합니다.

    - 기본 지표 `psnr_y`처럼 클수록 좋은 지표는 `mean_delta`가 가장 큰 영상이 성공(상위), 가장 작은 영상이 실패(하위)다.
    - `lpips`처럼 낮을수록 좋은 지표는 `mean_delta`가 가장 음수(작은)인 영상이 성공, 가장 큰 영상이 실패다.
    - 동점 시 `image_id` 오름차순으로 결정적으로 정렬한다.
    """
    if metric not in METRICS:
        raise ValueError(f"알 수 없는 지표입니다: {metric!r} (가능: {METRICS})")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k는 1 이상의 정수여야 합니다: {top_k!r}")

    table = _to_per_image_summary_table(
        per_image_df,
        config_id,
        summary_df=summary_df,
        expected_seeds=expected_seeds,
        metric=metric,
        baseline_config_id=baseline_config_id,
    )
    if datasets is not None:
        table = table[table["dataset"].isin(list(datasets))]
    pools = sorted(table["patch_pool_id"].unique())
    if len(pools) > 1:
        raise ValueError(f"서로 다른 패치 풀 {pools}이 섞여 있어 사례 영상을 한꺼번에 고를 수 없습니다.")
    # +inf/-inf(완전 복원 또는 베이스라인 완전 복원 대비 차이)는 순위를 매길 수 있으므로 유지하고,
    # +inf - +inf 등으로 정의되지 않은 NaN만 제외한다. 순위를 매길 수 있는 후보가 없으면 빈 결과를 반환한다.
    table = table[table["mean_delta"].notna()].copy()
    if table.empty:
        return {}

    direction = -1.0 if metric in LOWER_IS_BETTER_METRICS else 1.0
    table["_score"] = direction * table["mean_delta"].astype(float)

    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    dataset_order = list(datasets) if datasets is not None else sorted(table["dataset"].unique())
    for dataset in dataset_order:
        group = table[table["dataset"] == dataset]
        if group.empty:
            continue
        best = group.sort_values(["_score", "image_id"], ascending=[False, True], kind="mergesort").head(top_k)
        worst = group.sort_values(["_score", "image_id"], ascending=[True, True], kind="mergesort").head(top_k)
        selected[dataset] = {
            "success": _case_study_records(best, metric),
            "failure": _case_study_records(worst, metric),
        }
    return selected


@dataclass(frozen=True)
class _AblationContrast:
    contrast_id: str
    description: str
    target_config_id: str
    reference_config_id: str
    required_core: bool


_ABLATION_CONTRASTS: tuple[_AblationContrast, ...] = (
    _AblationContrast("A_rgb_context", "A 효과 (RGB 맥락: +A − 베이스라인)", "A", "baseline", True),
    _AblationContrast("A_y_context", "A 효과 (Y 맥락: +A+B − +B)", "AB", "B", True),
    _AblationContrast("B_alignment", "B 정렬 효과 (목표-지표 정렬: B′ − 베이스라인)", "Bp", "baseline", True),
    _AblationContrast("B_input_capacity", "B 입력·용량 효과 (+B − B′)", "B", "Bp", True),
    _AblationContrast("B_total", "B 전체 효과 (+B − 베이스라인)", "B", "baseline", True),
    _AblationContrast("AB_total", "핵심 조합 (+A+B − 베이스라인)", "AB", "baseline", True),
    _AblationContrast("C_total", "탐색 개선 (+C − 베이스라인)", "C", "baseline", False),
    _AblationContrast("A_in_C_context", "A 효과 (C 맥락: +A+C − +C)", "AC", "C", False),
    _AblationContrast("B_in_C_context", "B 효과 (C 맥락: +B+C − +C)", "BC", "C", False),
    _AblationContrast("C_in_AB_context", "C 추가 효과 (+A+B+C − +A+B)", "ABC", "AB", False),
)
_ABLATION_COLUMNS = [
    "patch_pool_id", "dataset", "contrast", "description",
    "target_config_id", "reference_config_id", "metric",
    "target_mean", "reference_mean", "delta", "seed_delta_std",
    "delta_sign", "delta_sign_consistent", "delta_exceeds_seed_delta_std",
    "delta_exceeds_seed_delta_std_display", "delta_direction",
    "n_paired_seeds", "paired_seeds", "is_complete",
    "target_run_ids", "reference_run_ids",
]  # fmt: skip


def _completeness_label(row: pd.Series) -> str:
    if bool(row["is_complete"]):
        base_ok = row.get("baseline_is_complete", pd.NA)
        if not pd.isna(base_ok) and not bool(base_ok):
            return "완료 (베이스라인 미완료)"
        return "완료"
    parts: list[str] = []
    if str(row.get("missing_seeds", "")).strip():
        parts.append(f"누락:{row['missing_seeds']}")
    if str(row.get("incomplete_seeds", "")).strip():
        parts.append(f"미완료:{row['incomplete_seeds']}")
    detail = ", ".join(parts) if parts else "부족"
    return f"미완료 ({detail})"


def _exceeds_display(metric: str, value: Any) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "—"
    if pd.isna(value):
        return "—"
    return "예" if bool(value) else "아니오"


def format_result_report_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """`compute_result_tables()['results']`를 보고서·발표용 결과 표 형식으로 정리합니다.

    - LPIPS는 Δ 부호를 그대로 두고 `delta_direction='음수가 개선'`을 표시하며,
      `delta_exceeds_seed_delta_std_display` 칸은 `—`로 둔다(각주 대상).
    - 미완료 구성(시드 3개 미만 또는 `status=incomplete`)은 `completeness` 열에 사유와 함께 명시한다.
    """
    out = results_df.copy()
    out["delta_direction"] = out["metric"].map(lambda m: "음수가 개선" if m in LOWER_IS_BETTER_METRICS else "양수가 개선")
    out["delta_exceeds_seed_delta_std_display"] = [
        _exceeds_display(m, v) for m, v in zip(out["metric"], out["delta_exceeds_seed_delta_std"])
    ]
    out["completeness"] = [_completeness_label(row) for _, row in out.iterrows()]
    return out


def build_ablation_table(summary_df: pd.DataFrame, *, expected_seeds: Sequence[int]) -> pd.DataFrame:
    """Ablation 표를 조립합니다 (SPEC US63, ADR-0004).

    - 핵심 분해: A 효과(RGB 맥락 `A − baseline`, Y 맥락 `AB − B`), B 효과(정렬 `Bp − baseline`,
      입력·용량 `B − Bp`, 전체 `B − baseline`), `AB − baseline`.
    - C 및 승격 구성(`AC`, `BC`, `ABC`)은 실행 기록이 있을 때 함께 싣는다.
    """
    expected, summary_df = _validate_summary_input(summary_df, expected_seeds)
    rows: list[dict[str, Any]] = []

    for (pool, dataset), block in summary_df.groupby(["patch_pool_id", "dataset"]):
        block_configs = set(block["config_id"].unique())
        complete_by_cfg: dict[str, pd.DataFrame] = {}
        completeness_by_cfg: dict[str, bool] = {}
        for cfg, cfg_rows in block.groupby("config_id"):
            complete_flag, _, _ = _is_complete(cfg_rows, expected, trainable=(cfg != BICUBIC_CONFIG_ID))
            completeness_by_cfg[cfg] = complete_flag
            complete_by_cfg[cfg] = cfg_rows[cfg_rows["status"] == "complete"].set_index("seed")

        for contrast in _ABLATION_CONTRASTS:
            if not contrast.required_core and (
                contrast.target_config_id not in block_configs or contrast.reference_config_id not in block_configs
            ):
                continue
            target_df = complete_by_cfg.get(
                contrast.target_config_id, pd.DataFrame(columns=block.columns).set_index("seed")
            )
            ref_df = complete_by_cfg.get(
                contrast.reference_config_id, pd.DataFrame(columns=block.columns).set_index("seed")
            )
            is_complete = bool(
                completeness_by_cfg.get(contrast.target_config_id, False)
                and completeness_by_cfg.get(contrast.reference_config_id, False)
            )
            for metric in METRICS:
                stats = _paired_contrast_stats(target_df, ref_df, expected, metric, pair_seeds=True)
                rows.append(
                    {
                        "patch_pool_id": pool,
                        "dataset": dataset,
                        "contrast": contrast.contrast_id,
                        "description": contrast.description,
                        "target_config_id": contrast.target_config_id,
                        "reference_config_id": contrast.reference_config_id,
                        "metric": metric,
                        "target_mean": stats["target_mean"],
                        "reference_mean": stats["ref_mean"],
                        "delta": stats["delta"],
                        "seed_delta_std": stats["seed_delta_std"],
                        "delta_sign": stats["delta_sign"],
                        "delta_sign_consistent": stats["delta_sign_consistent"],
                        "delta_exceeds_seed_delta_std": stats["delta_exceeds_seed_delta_std"],
                        "delta_exceeds_seed_delta_std_display": _exceeds_display(
                            metric, stats["delta_exceeds_seed_delta_std"]
                        ),
                        "delta_direction": "음수가 개선" if metric in LOWER_IS_BETTER_METRICS else "양수가 개선",
                        "n_paired_seeds": len(stats["paired"]),
                        "paired_seeds": _join_sorted(stats["paired"]),
                        "is_complete": is_complete,
                        "target_run_ids": _join_sorted(target_df["run_id"]) if len(target_df) else "",
                        "reference_run_ids": _join_sorted(ref_df["run_id"]) if len(ref_df) else "",
                    }
                )

    out = pd.DataFrame(rows, columns=_ABLATION_COLUMNS)
    for col in ("delta_sign_consistent", "delta_exceeds_seed_delta_std"):
        out[col] = out[col].astype("boolean")
    return out


def build_progress_table(
    summary_df: pd.DataFrame,
    *,
    expected_configs: Sequence[str],
    expected_seeds: Sequence[int],
    promoted_configs: Sequence[str] = (),
) -> pd.DataFrame:
    """구성 × 시드 진행 현황 표를 생성합니다 (SPEC US76)."""
    expected = sorted(set(int(s) for s in expected_seeds))
    if not expected:
        raise ValueError("expected_seeds가 비어 있습니다.")
    if not summary_df.empty:
        validate_unique_runs(summary_df)
        _check_statuses(summary_df, "summary_df")

    promoted_set = set(promoted_configs)
    config_order = list(expected_configs)
    if not summary_df.empty:
        for cfg in sorted(summary_df["config_id"].unique()):
            if cfg not in config_order:
                config_order.append(cfg)

    pools = sorted(summary_df["patch_pool_id"].unique()) if not summary_df.empty else [""]
    rows: list[dict[str, Any]] = []
    for pool in pools:
        pool_df = summary_df[summary_df["patch_pool_id"] == pool] if pool else summary_df
        for config_id in config_order:
            trainable = config_id != BICUBIC_CONFIG_ID
            promoted_only = config_id in promoted_set
            cfg_df = pool_df[pool_df["config_id"] == config_id]
            by_seed: dict[int, tuple[str, str]] = {}
            for seed_val, s_rows in cfg_df.groupby("seed"):
                statuses = sorted(set(s_rows["status"]))
                run_ids = sorted(set(s_rows["run_id"]))
                if len(statuses) != 1 or len(run_ids) != 1:
                    raise ValueError(f"({pool}, {config_id}, seed={seed_val})의 테스트셋 행들이 서로 다른 status/run_id를 가집니다.")
                by_seed[int(seed_val)] = (statuses[0], run_ids[0])

            row: dict[str, Any] = {
                "patch_pool_id": pool,
                "config_id": config_id,
                "trainable": trainable,
                "promoted_only": promoted_only,
            }
            complete_count = 0
            for seed in expected:
                if seed in by_seed:
                    st, rid = by_seed[seed]
                    row[f"seed_{seed}"] = f"{st} ({rid})"
                    if st == "complete":
                        complete_count += 1
                elif not trainable and len(by_seed) > 0:
                    row[f"seed_{seed}"] = "-"
                else:
                    row[f"seed_{seed}"] = "missing"
            expected_count = len(expected) if trainable else 1
            if len(by_seed) == 0:
                status = "보류(미승격)" if promoted_only else "미실행"
            elif complete_count >= expected_count:
                status = "완료"
            else:
                status = "미완료"
            row["complete_count"] = complete_count
            row["expected_count"] = expected_count
            row["status"] = status
            row["run_ids"] = _join_sorted(cfg_df["run_id"].unique())
            rows.append(row)
    return pd.DataFrame(rows)


def build_training_time_table(
    summary_df: pd.DataFrame,
    train_logs: dict[str, dict[str, Any]] | None = None,
    *,
    max_hours: float | None = None,
) -> pd.DataFrame:
    """실행별 학습 시간·파라미터·추론 시간 표를 생성합니다 (SPEC US65, ADR-0001)."""
    validate_unique_runs(summary_df)
    _check_statuses(summary_df, "summary_df")
    validated = _as_valid_numbers(
        summary_df, ("seed", "params", "infer_ms_net", "infer_ms_total", "train_minutes"), "summary_df"
    )
    train_logs = train_logs or {}
    rows: list[dict[str, Any]] = []
    for run_id, group in validated.groupby("run_id", sort=True):
        first = group.iloc[0]
        log = train_logs.get(str(run_id), {})
        losses = log.get("epoch_losses") or []
        train_minutes = float(first["train_minutes"])
        status = str(first["status"])
        within_limit: Any = pd.NA
        if max_hours is not None:
            within_limit = bool(status == "complete" and train_minutes <= float(max_hours) * 60.0)
        rows.append(
            {
                "patch_pool_id": str(first["patch_pool_id"]),
                "config_id": str(first["config_id"]),
                "seed": int(first["seed"]),
                "run_id": str(run_id),
                "status": status,
                "epochs_done": log.get("epochs_done", pd.NA),
                "epochs_total": log.get("epochs_total", pd.NA),
                "train_minutes": train_minutes,
                "within_max_hours": within_limit,
                "params": int(first["params"]),
                "infer_ms_net": float(first["infer_ms_net"]),
                "infer_ms_total": float(first["infer_ms_total"]),
                "final_epoch_loss": float(losses[-1]) if losses else math.nan,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["patch_pool_id", "config_id", "seed", "run_id"]).reset_index(drop=True)
    return df


def build_b_pre_rgb_diagnostics_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """B 계열 RGB 복원 전 Y 기준 PSNR(Y)(`psnr_y_pre_rgb`)과 공통 경로 `psnr_y`의 차이를 요약합니다 (ADR-0003)."""
    _check_statuses(summary_df, "summary_df")
    validated = _as_valid_numbers(
        summary_df, ("psnr_y",), "summary_df", optional=("psnr_y_pre_rgb",), allow_posinf=_PSNR_COLUMNS
    )
    complete = validated[(validated["status"] == "complete") & validated["psnr_y_pre_rgb"].notna()].copy()
    if complete.empty:
        return pd.DataFrame(
            columns=[
                "patch_pool_id", "dataset", "config_id", "n_seeds",
                "mean_psnr_y", "mean_psnr_y_pre_rgb", "diff_pre_minus_post", "run_ids",
            ]
        )
    complete["diff_pre_minus_post"] = complete["psnr_y_pre_rgb"] - complete["psnr_y"]
    rows: list[dict[str, Any]] = []
    for (pool, dataset, config_id), group in complete.groupby(["patch_pool_id", "dataset", "config_id"], sort=True):
        rows.append(
            {
                "patch_pool_id": pool,
                "dataset": dataset,
                "config_id": config_id,
                "n_seeds": len(group),
                "mean_psnr_y": float(group["psnr_y"].mean()),
                "mean_psnr_y_pre_rgb": float(group["psnr_y_pre_rgb"].mean()),
                "diff_pre_minus_post": float(group["diff_pre_minus_post"].mean()),
                "run_ids": _join_sorted(group["run_id"]),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FAIR_CONFIG = DEFAULT_REPO_ROOT / "configs" / "fair_comparison.yaml"
DEFAULT_EXPERIMENTS_CONFIG = DEFAULT_REPO_ROOT / "configs" / "experiments.yaml"
DEFAULT_DISTRIBUTION_DATASETS = ("BSD100", "Urban100")


def _load_experiments_metadata(path: Path) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    configs = data["configs"]
    order = list(configs.keys())
    promoted = [cfg for cfg, spec in configs.items() if spec.get("promoted_only", False)]
    return order, promoted, configs


def _load_train_logs(runs_dir: Path, run_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    logs: dict[str, dict[str, Any]] = {}
    for run_id in sorted(set(run_ids)):
        path = runs_dir / run_id / "train_log.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}는 JSON 객체여야 합니다.")
        logs[run_id] = payload
    return logs


def _format_cell(value: Any, column: str) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "예" if bool(value) else "아니오"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        if column in ("train_minutes", "infer_ms_net", "infer_ms_total"):
            return f"{float(value):.2f}"
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def _dataframe_to_markdown(df: pd.DataFrame, title: str, footnotes: Sequence[str] = ()) -> str:
    lines = [f"# {title}", ""]
    columns = list(df.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col], col) for col in columns) + " |")
    if footnotes:
        lines.append("")
        for note in footnotes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _write_table_pair(tables_dir: Path, stem: str, df: pd.DataFrame, title: str, footnotes: Sequence[str] = ()) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{stem}.csv", index=False)
    md_df = df.copy()
    if "delta_exceeds_seed_delta_std_display" in md_df.columns and "delta_exceeds_seed_delta_std" in md_df.columns:
        md_df["delta_exceeds_seed_delta_std"] = md_df["delta_exceeds_seed_delta_std_display"]
        md_df = md_df.drop(columns=["delta_exceeds_seed_delta_std_display"])
    (tables_dir / f"{stem}.md").write_text(_dataframe_to_markdown(md_df, title, footnotes), encoding="utf-8")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _plot_per_image_delta_distributions(
    per_image_summary: pd.DataFrame,
    figures_dir: Path,
    *,
    config_order: Sequence[str],
    datasets: Sequence[str] = DEFAULT_DISTRIBUTION_DATASETS,
    metric: str = "psnr_y",
    multi_pool: bool | None = None,
) -> list[str]:
    plt = _pyplot()
    created: list[str] = []
    if per_image_summary.empty:
        return created

    sub = per_image_summary[per_image_summary["metric"] == metric]
    pools = sorted(sub["patch_pool_id"].unique())
    is_multi_pool = (len(pools) > 1) if multi_pool is None else multi_pool
    for pool in pools:
        pool_df = sub[sub["patch_pool_id"] == pool]
        pool_prefix = f"{pool}_" if is_multi_pool and pool else ""
        for dataset in datasets:
            ds_df = pool_df[pool_df["dataset"] == dataset]
            if ds_df.empty:
                continue
            configs = [c for c in config_order if c in set(ds_df["config_id"])]
            if not configs:
                configs = sorted(ds_df["config_id"].unique())
            series_list = [
                ds_df.loc[ds_df["config_id"] == cfg, "mean_delta"].dropna().to_numpy(dtype=float) for cfg in configs
            ]
            series_list = [s[np.isfinite(s)] for s in series_list]
            if not any(len(s) for s in series_list):
                continue

            fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(configs) + 1.5), 4.5))
            ax.boxplot(series_list, showfliers=False)
            ax.set_xticks(range(1, len(configs) + 1), configs)
            rng = np.random.default_rng(0)
            for idx, (cfg, vals) in enumerate(zip(configs, series_list), start=1):
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(np.full(len(vals), idx) + jitter, vals, s=14, alpha=0.55)
                win_frac = float(np.mean(vals < 0 if metric in LOWER_IS_BETTER_METRICS else vals > 0))
                ax.text(idx, float(np.max(vals)), f"{win_frac:.0%}", ha="center", va="bottom", fontsize=8)
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
            ax.set_ylabel(f"Per-image delta ({metric})")
            title_pool = f" [{pool}]" if is_multi_pool and pool else ""
            ax.set_title(f"{dataset}: Per-image {metric} delta vs baseline{title_pool}")
            out_path = figures_dir / f"per_image_delta_distribution_{pool_prefix}{dataset}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            created.append(out_path.name)
    return created


def _plot_learning_curves(
    train_logs: dict[str, dict[str, Any]],
    figures_dir: Path,
    *,
    config_order: Sequence[str],
    run_to_pool: dict[str, str] | None = None,
    multi_pool: bool | None = None,
) -> list[str]:
    plt = _pyplot()
    by_pool: dict[str, dict[str, list[list[float]]]] = {}
    for run_id, log in train_logs.items():
        cfg = log.get("config_id")
        losses = log.get("epoch_losses")
        if cfg and isinstance(losses, list) and losses:
            pool = str(log.get("patch_pool_id") or (run_to_pool or {}).get(str(run_id), ""))
            by_pool.setdefault(pool, {}).setdefault(str(cfg), []).append([float(x) for x in losses])
    if not by_pool:
        return []

    is_multi_pool = (len(by_pool) > 1) if multi_pool is None else multi_pool
    created: list[str] = []
    for pool in sorted(by_pool):
        by_cfg = by_pool[pool]
        configs = [c for c in config_order if c in by_cfg] + sorted(c for c in by_cfg if c not in config_order)
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for cfg in configs:
            curves = by_cfg[cfg]
            max_len = max(len(c) for c in curves)
            padded = np.full((len(curves), max_len), np.nan, dtype=float)
            for i, c in enumerate(curves):
                padded[i, : len(c)] = c
            epochs = np.arange(1, max_len + 1)
            mean_curve = np.nanmean(padded, axis=0)
            line = ax.plot(epochs, mean_curve, label=cfg, linewidth=1.8)[0]
            if len(curves) > 1:
                ax.fill_between(
                    epochs, np.nanmin(padded, axis=0), np.nanmax(padded, axis=0), color=line.get_color(), alpha=0.15
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        title_pool = f" [{pool}]" if is_multi_pool and pool else ""
        ax.set_title(f"Training Loss Curves (mean across seeds){title_pool}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        out_name = f"learning_curves_{pool}.png" if is_multi_pool and pool else "learning_curves.png"
        out_path = figures_dir / out_name
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(out_name)
    return created


def _pick_canonical_runs(summary_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """패치 풀·구성별 대표 시각 비교에 쓸 단일 완료 실행 ID를 고른다(status=complete만, 시드 오름차순)."""
    chosen: dict[str, dict[str, str]] = {}
    complete_df = summary_df[summary_df["status"] == "complete"]
    for (pool, config_id), group in complete_df.groupby(["patch_pool_id", "config_id"], sort=True):
        first = group.sort_values(["seed", "run_id"]).iloc[0]
        chosen.setdefault(str(pool), {})[str(config_id)] = str(first["run_id"])
    return chosen


def _load_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _split_three_panels(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = strip.shape[1] // 3
    return strip[:, :width], strip[:, width : 2 * width], strip[:, 2 * width : 3 * width]


@dataclass(frozen=True)
class _ComparisonPanel:
    label: str
    image: np.ndarray
    roi: np.ndarray | None = None


def _image_to_shaved_tensor(img: np.ndarray, shave_border: int) -> Any:
    import torch

    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    if shave_border > 0 and min(img.shape[:2]) > 2 * shave_border:
        return tensor[..., shave_border:-shave_border, shave_border:-shave_border]
    return tensor


def _render_multi_config_panels(
    panels: list[_ComparisonPanel],
    hr_rgb: np.ndarray,
    hr_roi: np.ndarray | None,
    *,
    shave_border: int,
    comparison_path: Path,
    error_maps_path: Path,
    title: str,
) -> None:
    from sr_pipeline.diagnosis.diagnostics import compute_error_map

    plt = _pyplot()
    n_cols = len(panels) + 1
    has_roi = hr_roi is not None and all(p.roi is not None for p in panels)
    n_rows = 2 if has_roi else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows), squeeze=False)
    for col, panel in enumerate(panels):
        axes[0, col].imshow(panel.image)
        axes[0, col].set_title(panel.label, fontsize=9)
        axes[0, col].axis("off")
        if has_roi and panel.roi is not None:
            axes[1, col].imshow(panel.roi)
            axes[1, col].set_title(f"{panel.label} ROI", fontsize=8)
            axes[1, col].axis("off")
    axes[0, -1].imshow(hr_rgb)
    axes[0, -1].set_title("HR", fontsize=9)
    axes[0, -1].axis("off")
    if has_roi:
        axes[1, -1].imshow(hr_roi)
        axes[1, -1].set_title("HR ROI", fontsize=8)
        axes[1, -1].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(comparison_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    hr_t = _image_to_shaved_tensor(hr_rgb, shave_border)
    err_maps: list[tuple[str, np.ndarray]] = []
    vmax = 0.0
    for panel in panels:
        sr_t = _image_to_shaved_tensor(panel.image, shave_border)
        emap = compute_error_map(sr_t, hr_t).numpy()
        vmax = max(vmax, float(emap.max()) if emap.size else 0.0)
        err_maps.append((panel.label, emap))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, len(err_maps), figsize=(2.8 * len(err_maps), 3.0), squeeze=False)
    last_im = None
    for col, (label, emap) in enumerate(err_maps):
        last_im = axes[0, col].imshow(emap, cmap="inferno", vmin=0.0, vmax=vmax)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0, :].tolist(), label="|Y_SR - Y_HR| (8-bit)", shrink=0.85)
    fig.suptitle(f"{title} (Y Error Maps)", fontsize=10)
    fig.savefig(error_maps_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_representative_comparisons(
    runs_dir: Path,
    figures_dir: Path,
    summary_df: pd.DataFrame,
    per_image_df: pd.DataFrame,
    *,
    config_order: Sequence[str],
    shave_border: int,
) -> list[str]:
    canonical_by_pool = _pick_canonical_runs(summary_df)
    if not canonical_by_pool:
        return []
    psnr_lookup = {
        (row.run_id, row.dataset, row.image_id): float(row.psnr_y)
        for row in per_image_df.itertuples(index=False)
    }
    multi_pool = len(canonical_by_pool) > 1
    created: list[str] = []
    for pool, canonical in sorted(canonical_by_pool.items()):
        ordered_configs = [c for c in config_order if c in canonical] + sorted(
            c for c in canonical if c not in config_order
        )
        # (dataset, image_id, kind) 목록을 저장된 대표 영상에서 수집
        targets: set[tuple[str, str, str]] = set()
        for run_id in canonical.values():
            img_root = runs_dir / run_id / "images"
            if not img_root.is_dir():
                continue
            for path in img_root.glob("*/*.png"):
                if path.stem.endswith("_roi"):
                    continue
                for kind in ("median", "worst"):
                    suffix = f"_{kind}"
                    if path.stem.endswith(suffix):
                        targets.add((path.parent.name, path.stem[: -len(suffix)], kind))

        pool_prefix = f"{pool}_" if multi_pool else ""
        for dataset, image_id, kind in sorted(targets):
            panels: list[_ComparisonPanel] = []
            hr_rgb: np.ndarray | None = None
            hr_roi: np.ndarray | None = None
            bicubic_added = False
            for cfg in ordered_configs:
                run_id = canonical[cfg]
                img_path = runs_dir / run_id / "images" / dataset / f"{image_id}_{kind}.png"
                if not img_path.is_file():
                    continue
                before, after, hr = _split_three_panels(_load_rgb_u8(img_path))
                roi_path = img_path.with_name(f"{image_id}_{kind}_roi.png")
                before_roi, after_roi, hr_roi_cand = (
                    _split_three_panels(_load_rgb_u8(roi_path)) if roi_path.is_file() else (None, None, None)
                )
                if hr_rgb is None:
                    hr_rgb, hr_roi = hr, hr_roi_cand
                if not bicubic_added and cfg != BICUBIC_CONFIG_ID:
                    panels.append(_ComparisonPanel("bicubic", before, before_roi))
                    bicubic_added = True
                psnr_val = psnr_lookup.get((run_id, dataset, image_id))
                label = f"{cfg}\n({psnr_val:.2f} dB)" if psnr_val is not None else cfg
                panels.append(_ComparisonPanel(label, after, after_roi))
                if cfg == BICUBIC_CONFIG_ID:
                    bicubic_added = True
            if hr_rgb is None or not panels:
                continue
            comp_name = f"comparison_{pool_prefix}{dataset}_{image_id}_{kind}.png"
            err_name = f"error_maps_{pool_prefix}{dataset}_{image_id}_{kind}.png"
            _render_multi_config_panels(
                panels,
                hr_rgb,
                hr_roi,
                shave_border=shave_border,
                comparison_path=figures_dir / comp_name,
                error_maps_path=figures_dir / err_name,
                title=f"{dataset} {image_id} ({kind})" if not multi_pool else f"{pool} {dataset} {image_id} ({kind})",
            )
            created.extend([comp_name, err_name])
    return created


def _find_weights_path(runs_dir: Path, checkpoint_dir: Path | None, run_id: str) -> Path | None:
    if checkpoint_dir is not None:
        candidate = Path(checkpoint_dir) / run_id / "weights.pt"
        if candidate.is_file():
            return candidate
    fallback = runs_dir / run_id / "weights.pt"
    return fallback if fallback.is_file() else None


def _format_signed_delta(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.3f}"


def _render_case_study_figures(
    runs_dir: Path,
    figures_dir: Path,
    case_studies_df: pd.DataFrame,
    per_image_seed_deltas: pd.DataFrame,
    *,
    data_dir: Path,
    checkpoint_dir: Path | None,
    experiments: dict[str, dict[str, Any]],
    scale: int,
    shave_border: int,
    multi_pool: bool,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if case_studies_df.empty:
        return [], [], []
    import torch

    from sr_pipeline.common.bicubic import bicubic_resize
    from sr_pipeline.data.dataset import TestsetDataset
    from sr_pipeline.evaluation.evaluator import ROI_SIZE_HR, ROI_ZOOM, _display_u8, _roi_origin
    from sr_pipeline.models.espcn import ESPCN

    datasets_cache: dict[str, tuple[TestsetDataset, dict[str, int]]] = {}
    models_cache: dict[str, ESPCN] = {}

    def _get_model(cfg: str, run_id: str) -> ESPCN | None:
        if run_id in models_cache:
            return models_cache[run_id]
        weights_path = _find_weights_path(runs_dir, checkpoint_dir, run_id)
        if weights_path is None or cfg not in experiments:
            return None
        sw = experiments[cfg]["switches"]
        model = ESPCN(use_y_channel=bool(sw["B"]), use_global_residual=bool(sw["C"]))
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()
        models_cache[run_id] = model
        return model

    created: list[str] = []
    rendered_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    for row in case_studies_df.itertuples(index=False):
        pool = str(row.patch_pool_id)
        ds_name, image_id, cfg, case_type = str(row.dataset), str(row.image_id), str(row.config_id), str(row.case_type)
        metric = str(row.metric)
        paired_rows = per_image_seed_deltas[
            (per_image_seed_deltas["patch_pool_id"] == pool)
            & (per_image_seed_deltas["dataset"] == ds_name)
            & (per_image_seed_deltas["config_id"] == cfg)
            & (per_image_seed_deltas["image_id"] == image_id)
            & (per_image_seed_deltas["metric"] == metric)
        ].sort_values("seed")

        chosen_pair: tuple[int, str, str, ESPCN, ESPCN] | None = None
        missing_details: list[str] = []
        for pair in paired_rows.itertuples(index=False):
            seed_val = int(pair.seed)
            t_run_id = str(pair.run_id)
            b_run_id = str(pair.baseline_run_id)
            t_model = _get_model(cfg, t_run_id)
            b_model = _get_model(BASELINE_CONFIG_ID, b_run_id)
            if t_model is not None and b_model is not None:
                chosen_pair = (seed_val, t_run_id, b_run_id, t_model, b_model)
                break
            missing_sides: list[str] = []
            if t_model is None:
                missing_sides.append(t_run_id)
            if b_model is None:
                missing_sides.append(b_run_id)
            missing_details.append(f"seed={seed_val}({', '.join(missing_sides)} 체크포인트 없음)")

        if chosen_pair is None:
            reason = (
                "짝지어진 시드 쌍의 체크포인트가 없습니다: " + "; ".join(missing_details)
                if missing_details
                else "Δ 계산에 참여한 시드 쌍을 찾지 못했습니다."
            )
            skipped_records.append(
                {
                    "patch_pool_id": pool,
                    "config_id": cfg,
                    "dataset": ds_name,
                    "image_id": image_id,
                    "case_type": case_type,
                    "reason": reason,
                }
            )
            continue

        if ds_name not in datasets_cache:
            try:
                ds = TestsetDataset(data_dir, ds_name)
            except ValueError as error:
                skipped_records.append(
                    {
                        "patch_pool_id": pool,
                        "config_id": cfg,
                        "dataset": ds_name,
                        "image_id": image_id,
                        "case_type": case_type,
                        "reason": str(error),
                    }
                )
                continue
            datasets_cache[ds_name] = (ds, {img_id: idx for idx, img_id in enumerate(ds.image_ids)})
        ds, id_map = datasets_cache[ds_name]
        if image_id not in id_map:
            skipped_records.append(
                {
                    "patch_pool_id": pool,
                    "config_id": cfg,
                    "dataset": ds_name,
                    "image_id": image_id,
                    "case_type": case_type,
                    "reason": f"테스트셋 {ds_name}에 영상 {image_id!r}가 없습니다.",
                }
            )
            continue

        seed_val, target_run_id, baseline_run_id, target_model, base_model = chosen_pair
        sample = ds[id_map[image_id]]
        lr, hr = sample["lr"], sample["hr"]
        hr_u8 = _display_u8(hr)
        height, width = hr.shape[-2:]
        size = min(ROI_SIZE_HR, height, width)
        top, left = _roi_origin(hr, size)

        def _zoom(arr: np.ndarray) -> np.ndarray:
            roi_window = arr[top : top + size, left : left + size]
            return np.repeat(np.repeat(roi_window, ROI_ZOOM, 0), ROI_ZOOM, 1)

        with torch.inference_mode():
            bic_u8 = _display_u8(bicubic_resize(lr, float(scale)))
            base_u8 = _display_u8(base_model(lr))
            target_u8 = _display_u8(target_model(lr))
            panels: list[_ComparisonPanel] = [
                _ComparisonPanel("bicubic", bic_u8, _zoom(bic_u8)),
                _ComparisonPanel(f"{BASELINE_CONFIG_ID}\n({baseline_run_id})", base_u8, _zoom(base_u8)),
                _ComparisonPanel(f"{cfg}\n({target_run_id})", target_u8, _zoom(target_u8)),
            ]

        pool_prefix = f"{pool}_" if multi_pool else ""
        comp_name = f"case_study_comparison_{pool_prefix}{cfg}_{ds_name}_{image_id}_{case_type}.png"
        err_name = f"case_study_error_maps_{pool_prefix}{cfg}_{ds_name}_{image_id}_{case_type}.png"
        delta_str = _format_signed_delta(float(row.mean_delta))
        _render_multi_config_panels(
            panels,
            hr_u8,
            _zoom(hr_u8),
            shave_border=shave_border,
            comparison_path=figures_dir / comp_name,
            error_maps_path=figures_dir / err_name,
            title=f"{ds_name} {image_id} ({cfg} {case_type}, seed={seed_val}, delta={delta_str} dB)",
        )
        created.extend([comp_name, err_name])
        rendered_records.append(
            {
                "comparison_figure": comp_name,
                "error_maps_figure": err_name,
                "patch_pool_id": pool,
                "config_id": cfg,
                "dataset": ds_name,
                "image_id": image_id,
                "case_type": case_type,
                "seed": seed_val,
                "run_id": target_run_id,
                "baseline_run_id": baseline_run_id,
            }
        )
    return created, rendered_records, skipped_records


def export_report_assets(
    runs_dir: Path,
    output_dir: Path,
    *,
    fair_config_path: Path = DEFAULT_FAIR_CONFIG,
    experiments_config_path: Path = DEFAULT_EXPERIMENTS_CONFIG,
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    allowed_run_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """표(CSV/Markdown), 분포 그림, 다중 구성 비교 그림, 학습 곡선, 진행 현황 표를 내보냅니다."""
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    fair = load_fair_comparison(fair_config_path)
    expected_seeds = list(fair["fixed_rules"]["seeds"])
    scale = int(fair["fixed_rules"]["scale"])
    shave_border = int(fair["fixed_rules"]["shave_border"])
    max_hours = float(fair["fixed_rules"]["max_hours"])
    config_order, promoted_configs, experiments = _load_experiments_metadata(experiments_config_path)

    summary_df, per_image_df = load_run_records(runs_dir, allowed_run_ids=allowed_run_ids)
    validate_unique_runs(summary_df)
    result_tables = compute_result_tables(summary_df, expected_seeds=expected_seeds)
    formatted_results = format_result_report_table(result_tables["results"])
    ablation_df = build_ablation_table(summary_df, expected_seeds=expected_seeds)
    per_image_tables = compute_per_image_deltas(summary_df, per_image_df, expected_seeds=expected_seeds)
    train_logs = _load_train_logs(runs_dir, summary_df["run_id"].unique())
    training_times_df = build_training_time_table(summary_df, train_logs=train_logs, max_hours=max_hours)
    progress_df = build_progress_table(
        summary_df,
        expected_configs=config_order,
        expected_seeds=expected_seeds,
        promoted_configs=promoted_configs,
    )
    b_pre_rgb_df = build_b_pre_rgb_diagnostics_table(summary_df)

    case_rows: list[dict[str, Any]] = []
    per_image_summary = per_image_tables["per_image_summary"]
    target_case_configs = [
        cfg
        for cfg in config_order
        if cfg not in (BASELINE_CONFIG_ID, BICUBIC_CONFIG_ID) and cfg in set(per_image_summary["config_id"])
    ]
    for pool in sorted(per_image_summary["patch_pool_id"].unique()) if not per_image_summary.empty else []:
        pool_summary = summary_df[summary_df["patch_pool_id"] == pool]
        pool_per_img = per_image_df[per_image_df["run_id"].isin(set(pool_summary["run_id"]))]
        for cfg in target_case_configs:
            if cfg not in set(per_image_summary.loc[per_image_summary["patch_pool_id"] == pool, "config_id"]):
                continue
            selected = select_case_study_images(
                pool_per_img, cfg, summary_df=pool_summary, expected_seeds=expected_seeds, metric="psnr_y"
            )
            for ds_name, kinds in selected.items():
                for case_type in ("success", "failure"):
                    for item in kinds[case_type]:
                        case_rows.append({"case_type": case_type, **item})
    case_studies_df = pd.DataFrame(
        case_rows,
        columns=[
            "patch_pool_id", "config_id", "dataset", "case_type", "rank", "image_id",
            "metric", "mean_delta", "n_seeds", "run_ids", "baseline_run_ids",
        ],
    )

    lpips_note = (
        "LPIPS는 낮을수록 좋으므로 Δ는 그대로 표시하되 음수가 개선이며, "
        "`delta_exceeds_seed_delta_std` 칸은 `—`로 표기합니다."
    )
    incomplete_note = "시드 3개 미만 또는 `status=incomplete` 구성은 완료 시드 수치만 계산하고 미완료로 표시합니다."

    staging_root = Path(tempfile.mkdtemp(prefix="aggregate_staging_"))
    try:
        tables_dir = staging_root / "tables"
        figures_dir = staging_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        _write_table_pair(tables_dir, "results", formatted_results, "실행 결과 요약 표", (lpips_note, incomplete_note))
        _write_table_pair(
            tables_dir,
            "seed_deltas",
            result_tables["seed_deltas"],
            "시드별 Δ 표",
            ("LPIPS는 낮을수록 좋으므로 Δ는 그대로 표시하되 음수가 개선입니다.",),
        )
        _write_table_pair(tables_dir, "ablation", ablation_df, "Ablation 분해 표", (lpips_note, incomplete_note))
        _write_table_pair(tables_dir, "case_studies", case_studies_df, "자동 선정 사례 영상 (영상별 Δ 상·하위)")
        _write_table_pair(tables_dir, "training_times", training_times_df, "학습 시간 및 추론 시간 표")
        _write_table_pair(tables_dir, "progress", progress_df, "구성 × 시드 진행 현황 표")
        _write_table_pair(
            tables_dir,
            "b_pre_rgb_diagnostics",
            b_pre_rgb_df,
            "B 계열 RGB 복원 전후 PSNR(Y) 진단 표",
            ("결과 표에는 공통 평가 경로(`psnr_y`) 값만 사용합니다.",),
        )
        per_image_tables["per_image_summary"].to_csv(tables_dir / "per_image_summary.csv", index=False)
        per_image_tables["per_image_seed_deltas"].to_csv(tables_dir / "per_image_seed_deltas.csv", index=False)

        multi_pool = summary_df["patch_pool_id"].nunique() > 1
        run_to_pool = {
            str(r.run_id): str(r.patch_pool_id)
            for r in summary_df[["run_id", "patch_pool_id"]].drop_duplicates().itertuples(index=False)
        }
        figures_created: list[str] = []
        figures_created += _plot_per_image_delta_distributions(
            per_image_summary, figures_dir, config_order=config_order, multi_pool=multi_pool
        )
        figures_created += _plot_learning_curves(
            train_logs, figures_dir, config_order=config_order, run_to_pool=run_to_pool, multi_pool=multi_pool
        )
        figures_created += _render_representative_comparisons(
            runs_dir,
            figures_dir,
            summary_df,
            per_image_df,
            config_order=config_order,
            shave_border=shave_border,
        )
        rendered_case_records: list[dict[str, Any]] = []
        skipped_case_records: list[dict[str, Any]] = []
        case_study_note = None
        if data_dir is not None:
            case_figs, rendered_case_records, skipped_case_records = _render_case_study_figures(
                runs_dir,
                figures_dir,
                case_studies_df,
                per_image_tables["per_image_seed_deltas"],
                data_dir=Path(data_dir),
                checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None,
                experiments=experiments,
                scale=scale,
                shave_border=shave_border,
                multi_pool=multi_pool,
            )
            figures_created += case_figs
            if skipped_case_records and not case_figs:
                case_study_note = "짝지어진 시드의 체크포인트(weights.pt)를 찾지 못해 사례 영상 비교·오차 지도를 생성하지 못했습니다."
            elif skipped_case_records:
                case_study_note = (
                    f"사례 영상 {len(skipped_case_records)}건은 짝지어진 시드 체크포인트 또는 테스트셋 영상을 찾지 못해 건너뛰었습니다."
                )
        elif not case_studies_df.empty:
            case_study_note = "--data-dir이 지정되지 않아 사례 영상 비교·오차 지도 그림을 건너뛰었습니다."
            skipped_case_records = [
                {
                    "patch_pool_id": str(row.patch_pool_id),
                    "config_id": str(row.config_id),
                    "dataset": str(row.dataset),
                    "image_id": str(row.image_id),
                    "case_type": str(row.case_type),
                    "reason": "--data-dir이 지정되지 않음",
                }
                for row in case_studies_df.itertuples(index=False)
            ]

        manifest: dict[str, Any] = {
            "run_ids": sorted(str(r) for r in summary_df["run_id"].unique()),
            "tables": sorted(p.name for p in tables_dir.iterdir() if p.is_file()),
            "figures": sorted(figures_created),
        }
        if rendered_case_records:
            manifest["case_study_figures"] = rendered_case_records
        if skipped_case_records:
            manifest["skipped_case_study_figures"] = skipped_case_records
        if case_study_note is not None:
            manifest["case_study_figures_skipped_reason"] = case_study_note
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("tables", "figures"):
            dest = output_dir / sub
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(staging_root / sub), str(dest))
        shutil.move(str(staging_root / "manifest.json"), str(output_dir / "manifest.json"))
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return manifest
