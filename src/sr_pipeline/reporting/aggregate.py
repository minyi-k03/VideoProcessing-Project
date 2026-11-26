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

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def load_run_records(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`runs_dir/<run_id>/{summary,per_image}.csv`를 모두 읽어 (summary, per_image)로 합칩니다.

    실행 폴더에 두 파일 중 하나라도 없거나, 파일에 행이 없거나, 필요한 열이 없거나,
    필수 값이 비었거나(빈 값은 `psnr_y_pre_rgb`만 허용), 허용되지 않은 수치(`+inf`는 PSNR 계열 열만 허용,
    `NaN`·`-inf`·비숫자는 모두 거부)가 있거나, `status`가 알 수 없는 값이면 집계 전에 `ValueError`로 멈춘다.
    중복 검사는 `validate_unique_runs`가 한다.
    "main에 병합된 기록만"이라는 조건은 T6.3 집계 스크립트(`scripts/aggregate_results.py`)에서 검증한다.
    """
    runs_dir = Path(runs_dir)
    run_dirs = (
        sorted({p.parent for name in (SUMMARY_FILENAME, PER_IMAGE_FILENAME) for p in runs_dir.glob(f"*/{name}")})
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
    expected = sorted(set(int(s) for s in expected_seeds))
    if not expected:
        raise ValueError("expected_seeds가 비어 있습니다.")
    validate_unique_runs(summary_df)
    _check_statuses(summary_df, "summary_df")
    summary_df = _as_valid_numbers(summary_df, METRICS, "summary_df", allow_posinf=_PSNR_COLUMNS)
    unexpected = sorted(set(int(s) for s in summary_df["seed"]) - set(expected))
    if unexpected:
        raise ValueError(f"fair_comparison의 시드 {expected}에 없는 시드가 있습니다: {unexpected}")

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
            paired_by_seed = trainable and has_delta
            paired = sorted(set(by_seed.index) & set(baseline_complete_rows.index)) if paired_by_seed else []
            all_seeds_paired = paired == expected

            for metric in METRICS:
                values = list(by_seed[metric])
                mean = float(np.mean(values)) if values else math.nan
                baseline_mean = (
                    float(np.mean(list(baseline_complete_rows[metric]))) if has_delta else math.nan
                )
                delta = (mean - baseline_mean) if has_delta else math.nan
                seed_deltas = [
                    float(by_seed.at[s, metric]) - float(baseline_complete_rows.at[s, metric]) for s in paired
                ]
                for s, d in zip(paired, seed_deltas):
                    seed_delta_rows.append(
                        {"patch_pool_id": pool, "dataset": dataset, "config_id": config_id, "metric": metric,
                         "seed": int(s), "delta": d,
                         "run_id": by_seed.at[s, "run_id"], "baseline_run_id": baseline_complete_rows.at[s, "run_id"]}
                    )  # fmt: skip
                sign = _delta_sign(seed_deltas)
                seed_delta_std = _sample_std(seed_deltas)
                consistent: Any = pd.NA
                exceeds: Any = pd.NA
                if all_seeds_paired and sign != "":
                    consistent = sign != "mixed"
                    if not math.isnan(seed_delta_std):
                        mean_seed_delta = float(np.mean(seed_deltas))
                        if math.isfinite(mean_seed_delta):
                            exceeds = mean_seed_delta > seed_delta_std

                result_rows.append(
                    {
                        "patch_pool_id": pool, "dataset": dataset, "config_id": config_id, "metric": metric,
                        "n_seeds": len(values), "seeds": _join_sorted(by_seed.index),
                        "mean": mean, "std": _sample_std(values), "delta": delta,
                        "seed_delta_std": seed_delta_std, "delta_sign": sign,
                        "delta_sign_consistent": consistent, "delta_exceeds_seed_delta_std": exceeds,
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


def select_case_study_images(per_image_df: Any, config_id: str) -> dict[str, Any]:
    """영상별 Δ의 상위·하위에서 성공·실패 사례 영상을 규칙으로 자동 선정합니다."""
    raise NotImplementedError("TODO(T6.3): Select top/bottom per-image delta cases objectively.")


def export_report_assets(runs_dir: Path, output_dir: Path) -> None:
    """표(CSV/Markdown), 분포 그림, 다중 구성 비교 그림, 학습 곡선, 진행 현황 표를 내보냅니다."""
    raise NotImplementedError("TODO(T6.3): Run full aggregation pipeline and write report assets.")
