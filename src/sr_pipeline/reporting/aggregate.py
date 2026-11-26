"""실행 기록 자동 집계 및 보고서/발표용 표·그림 생성 (SPEC.md / ADR-0003).

원칙:
- main에 병합된 실행 기록(`results/runs/`)만 입력으로 사용
- `(config_id, seed, patch_pool_id)` 중복 감지 시 오류
- 시드 3개 미만 또는 `incomplete` 포함 구성은 "미완료" 표시
- 평균 ± 표준편차, 베이스라인 대비 Δ, 시드별 Δ 부호 일관성, |mean Δ| > std(seed Δ) 판정
- Ablation 표(A: RGB·Y 맥락, B: 정렬 B′·입력/용량 B 분해), 영상별 Δ 분포, 사례 영상 자동 선정

이 파일의 구현 범위(T4.3 집계 핵심): `load_run_records`, `validate_unique_runs`, `compute_result_tables`.
Ablation 표 조립·영상별 Δ·사례 영상·그림·`export_report_assets`는 T6.3에서 구현한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
_REQUIRED_SUMMARY_VALUES = ("run_id", "config_id", "seed", "patch_pool_id", "dataset", "status", *METRICS)
_REQUIRED_PER_IMAGE_VALUES = ("run_id", "dataset", "image_id", *METRICS)

_RESULT_COLUMNS = [
    "patch_pool_id", "dataset", "config_id", "metric",
    "n_seeds", "seeds", "mean", "std", "delta",
    "seed_delta_std", "delta_sign", "delta_sign_consistent", "delta_exceeds_seed_delta_std",
    "is_complete", "missing_seeds", "incomplete_seeds", "baseline_is_complete",
]  # fmt: skip
_SEED_DELTA_COLUMNS = [*_GROUP_KEY, "metric", "seed", "delta"]


def _read_csv(path: Path) -> pd.DataFrame:
    # "none"(스위치 없음) 같은 문자열이 결측으로 바뀌지 않게 하고, 빈 칸만 결측으로 읽는다.
    # 식별자 열은 숫자처럼 보여도(예: 숫자만 있는 짧은 커밋) 문자열로 읽는다.
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={c: str for c in _STRING_COLUMNS})


def _check_no_blanks(df: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    """키·지표 열의 빈 칸은 groupby가 조용히 버리거나 평균에 NaN으로 번지므로 읽을 때 거부한다."""
    blank = [c for c in columns if df[c].isna().any()]
    if blank:
        raise ValueError(f"{source}에 빈 값이 있는 필수 열이 있습니다: {blank}")


def _check_columns(df: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}에 필요한 열이 없습니다: {missing}")


def load_run_records(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`runs_dir/<run_id>/{summary,per_image}.csv`를 모두 읽어 (summary, per_image)로 합칩니다.

    실행 폴더에 두 파일 중 하나라도 없거나, 필요한 열이 없거나, `status`가 알 수 없는 값이면
    집계 전에 `ValueError`로 멈춘다. 중복 검사는 `validate_unique_runs`가 한다.
    "main에 병합된 기록만"이라는 조건은 호출자가 `runs_dir`을 main 체크아웃에서 가리켜 지킨다(여기서 확인하지 않음).
    """
    runs_dir = Path(runs_dir)
    summary_paths = sorted(runs_dir.glob(f"*/{SUMMARY_FILENAME}")) if runs_dir.is_dir() else []
    if not summary_paths:
        raise ValueError(f"실행 기록이 없습니다(<실행 폴더>/{SUMMARY_FILENAME}를 찾지 못함): {runs_dir}")

    summaries: list[pd.DataFrame] = []
    per_images: list[pd.DataFrame] = []
    for summary_path in summary_paths:
        per_image_path = summary_path.with_name(PER_IMAGE_FILENAME)
        if not per_image_path.is_file():
            raise ValueError(f"{summary_path.parent}에 {PER_IMAGE_FILENAME}가 없습니다.")
        summary = _read_csv(summary_path)
        per_image = _read_csv(per_image_path)
        _check_columns(summary, _SUMMARY_COLUMNS, summary_path)
        _check_columns(per_image, _PER_IMAGE_COLUMNS, per_image_path)
        _check_no_blanks(summary, _REQUIRED_SUMMARY_VALUES, summary_path)
        _check_no_blanks(per_image, _REQUIRED_PER_IMAGE_VALUES, per_image_path)
        summaries.append(summary[list(_SUMMARY_COLUMNS)])
        per_images.append(per_image[list(_PER_IMAGE_COLUMNS)])

    summary_df = pd.concat(summaries, ignore_index=True)
    per_image_df = pd.concat(per_images, ignore_index=True)

    bad = sorted(set(summary_df["status"]) - set(_STATUSES))
    if bad:
        raise ValueError(f"summary의 status는 {_STATUSES} 중 하나여야 합니다: {bad}")
    orphan = sorted(set(per_image_df["run_id"]) - set(summary_df["run_id"]))
    if orphan:
        raise ValueError(f"summary에 없는 실행의 per_image 기록이 있습니다: {orphan}")
    return summary_df, per_image_df


def validate_unique_runs(summary_df: pd.DataFrame) -> None:
    """`(config_id, seed, patch_pool_id)` 조합이 둘 이상의 실행에 나타나면 `ValueError`를 냅니다.

    실행 1회는 테스트셋마다 summary 행을 하나씩 만들므로, 한 실행 안에서 같은 테스트셋 행이
    두 번 나오는 것도 중복이다. 커밋이 달라 `run_id`가 다른 재실행도 같은 조합이면 중복이다.
    """
    problems: list[str] = []
    flagged: set[tuple] = set()
    runs_per_key = summary_df.groupby(_RUN_KEY)["run_id"].nunique()
    for key, n_runs in runs_per_key.items():
        if n_runs > 1:
            flagged.add(key)
            run_ids = sorted(summary_df.loc[_matches(summary_df, _RUN_KEY, key), "run_id"].unique())
            problems.append(f"{dict(zip(_RUN_KEY, key))}: 실행 {run_ids}")
    rows_per_dataset = summary_df.groupby([*_RUN_KEY, "dataset"]).size()
    for key, n_rows in rows_per_dataset.items():
        if n_rows > 1 and key[:3] not in flagged:
            problems.append(f"{dict(zip([*_RUN_KEY, 'dataset'], key))}: summary 행 {n_rows}개")
    if problems:
        raise ValueError("(config_id, seed, patch_pool_id) 중복 실행 기록이 있습니다:\n  " + "\n  ".join(problems))


def _matches(df: pd.DataFrame, columns: Sequence[str], values: Sequence[Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in zip(columns, values):
        mask &= df[column] == value
    return mask


def _sample_std(values: Sequence[float]) -> float:
    """표본 표준편차(ddof=1). 값이 2개 미만이면 정의되지 않아 NaN."""
    return float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan


def _join_seeds(seeds: Sequence[int]) -> str:
    return ",".join(str(s) for s in sorted(seeds))


def compute_result_tables(summary_df: pd.DataFrame, *, expected_seeds: Sequence[int]) -> dict[str, pd.DataFrame]:
    """평균 ± 표준편차, Δ, 시드별 Δ 일관성, 미완료 표시를 계산합니다.

    `expected_seeds`는 `fair_comparison.yaml`의 `fixed_rules.seeds`를 호출자가 넘긴다(기본값 없음).
    (패치 풀, 테스트셋, 구성, 지표)마다 한 행인 `results`와, 시드별 Δ를 모두 담은 `seed_deltas`를 돌려준다.

    - `std`·`seed_delta_std`: 표본 표준편차(ddof=1), 값이 2개 미만이면 NaN.
    - `delta`: 같은 패치 풀·테스트셋·지표에서 (구성의 시드 평균) − (베이스라인의 시드 평균).
      베이스라인 자신, 베이스라인이 없는 경우는 NaN.
    - `delta_sign`: 같은 시드끼리 짝지은 시드별 Δ가 모두 양수면 `positive`, 모두 음수면 `negative`,
      그 외(0 포함)는 `mixed`. 베이스라인·Bicubic 하한·짝지을 시드가 없으면 빈 문자열.
    - `delta_exceeds_seed_delta_std`: |평균 시드별 Δ| > 시드별 Δ의 표준편차. 정의되지 않으면 `<NA>`.
    - `is_complete`: 기대 시드가 모두 있고 모두 `status=complete`. Bicubic 하한은 시드 하나면 충분하다.
    - `baseline_is_complete`: 같은 패치 풀·테스트셋의 베이스라인이 완료인지(Δ의 기준이 미완료인지 알리기 위함).
    """
    expected = sorted(set(int(s) for s in expected_seeds))
    if not expected:
        raise ValueError("expected_seeds가 비어 있습니다.")
    unexpected = sorted(set(int(s) for s in summary_df["seed"]) - set(expected))
    if unexpected:
        raise ValueError(f"fair_comparison의 시드 {expected}에 없는 시드가 있습니다: {unexpected}")
    validate_unique_runs(summary_df)

    result_rows: list[dict[str, Any]] = []
    seed_delta_rows: list[dict[str, Any]] = []

    for (pool, dataset), block in summary_df.groupby(["patch_pool_id", "dataset"]):
        baseline = block[block["config_id"] == BASELINE_CONFIG_ID]
        baseline_complete = None
        if len(baseline):
            baseline_complete, _, _ = _is_complete(baseline, expected, trainable=True)

        for config_id, config_rows in block.groupby("config_id"):
            trainable = config_id != BICUBIC_CONFIG_ID
            complete, missing, incomplete = _is_complete(config_rows, expected, trainable=trainable)
            compare = trainable and config_id != BASELINE_CONFIG_ID and len(baseline) > 0

            for metric in METRICS:
                by_seed = config_rows.set_index("seed")[metric].astype(float)
                values = list(by_seed.values)
                mean = float(np.mean(values))
                delta = math.nan
                sign = ""
                consistent: Any = pd.NA
                exceeds: Any = pd.NA
                seed_delta_std = math.nan

                if config_id == BICUBIC_CONFIG_ID and len(baseline):
                    delta = mean - float(baseline[metric].astype(float).mean())
                elif compare:
                    base_by_seed = baseline.set_index("seed")[metric].astype(float)
                    delta = mean - float(base_by_seed.mean())
                    paired = sorted(set(by_seed.index) & set(base_by_seed.index))
                    seed_deltas = [float(by_seed[s] - base_by_seed[s]) for s in paired]
                    for s, d in zip(paired, seed_deltas):
                        seed_delta_rows.append(
                            {"patch_pool_id": pool, "dataset": dataset, "config_id": config_id,
                             "metric": metric, "seed": int(s), "delta": d}
                        )  # fmt: skip
                    if seed_deltas:
                        if all(d > 0 for d in seed_deltas):
                            sign = "positive"
                        elif all(d < 0 for d in seed_deltas):
                            sign = "negative"
                        else:
                            sign = "mixed"
                        consistent = sign != "mixed"
                        seed_delta_std = _sample_std(seed_deltas)
                        if not math.isnan(seed_delta_std):
                            exceeds = abs(float(np.mean(seed_deltas))) > seed_delta_std

                result_rows.append(
                    {
                        "patch_pool_id": pool, "dataset": dataset, "config_id": config_id, "metric": metric,
                        "n_seeds": len(values), "seeds": _join_seeds(by_seed.index),
                        "mean": mean, "std": _sample_std(values), "delta": delta,
                        "seed_delta_std": seed_delta_std, "delta_sign": sign,
                        "delta_sign_consistent": consistent, "delta_exceeds_seed_delta_std": exceeds,
                        "is_complete": complete, "missing_seeds": _join_seeds(missing),
                        "incomplete_seeds": _join_seeds(incomplete),
                        "baseline_is_complete": baseline_complete if baseline_complete is not None else pd.NA,
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
