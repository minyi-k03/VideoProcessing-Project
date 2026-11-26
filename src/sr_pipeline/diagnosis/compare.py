"""두 진단 결과의 영상별 비교 (SPEC.md US57·83, Task T3.1).

파일럿 베이스라인 진단과 Bicubic 하한 진단처럼, 같은 테스트셋·같은 진단 설정으로 만든 두 진단 출력 폴더를
영상 단위로 맞대어 테스트셋별 차이를 표로 만든다. 개선 확정(ADR-0004)의 근거 수치는 이 표에서만 인용한다
(ADR-0006: 손으로 옮겨 적은 수치는 쓰지 않는다).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sr_pipeline.diagnosis.diagnostics import DIAGNOSIS_SETTINGS_FILENAME, PIXEL_MAX

PER_IMAGE_FILENAME = "per_image_diagnosis.csv"
# 두 진단이 같아야 비교가 성립하는 설정.
MATCHED_SETTINGS = ("edge_top_q", "shave_border", "datasets")
# 모델 − 기준 평균 차이를 내는 열.
DELTA_COLUMNS = (
    "psnr_y",
    "edge_psnr_y",
    "flat_psnr_y",
    "mse_cb",
    "mse_cr",
    "band_ratio_low",
    "band_ratio_mid",
    "band_ratio_high",
)
# 모델이 이긴 영상 비율을 내는 열과, 이긴 방향(+1: 클수록 좋음, -1: 작을수록 좋음).
WIN_COLUMNS = {"psnr_y": 1, "edge_psnr_y": 1, "flat_psnr_y": 1, "mse_cb": -1, "mse_cr": -1}
_KEYS = ["dataset", "image_id"]


def _load_diagnosis(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = Path(directory)
    settings_path = directory / DIAGNOSIS_SETTINGS_FILENAME
    per_image_path = directory / PER_IMAGE_FILENAME
    for path in (settings_path, per_image_path):
        if not path.is_file():
            raise FileNotFoundError(f"진단 출력 폴더에 {path.name}이(가) 없습니다: {directory}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path.name}은(는) 키-값 묶음이어야 합니다: {directory}")
    table = pd.read_csv(per_image_path, dtype={"image_id": str})
    missing = set(_KEYS) | set(DELTA_COLUMNS) | {"share_y"}
    missing -= set(table.columns)
    if missing:
        raise ValueError(f"{per_image_path.name}에 필요한 열이 없습니다: {sorted(missing)} ({directory})")
    if table.duplicated(_KEYS).any():
        raise ValueError(f"{per_image_path.name}에 같은 (dataset, image_id) 행이 여러 개 있습니다: {directory}")
    return table, settings


def _edge_error_share(edge_psnr_y: pd.Series, flat_psnr_y: pd.Series, edge_fraction: float) -> pd.Series:
    """영상별 Y 오차(제곱 오차 합) 중 에지 영역이 차지하는 비중.

    에지·평탄 PSNR(Y)을 MSE로 되돌리고 영역 화소 비율로 가중한다. 에지 영역 화소 수는 ceil(q%·N)이라
    비율을 q%로 둔 이 값은 근사다(테스트셋 영상 크기에서 차이는 화소 1개 이하).
    Y 오차가 0인 영상(에지·평탄 PSNR(Y) 모두 inf)은 비중이 정의되지 않아 NaN이다. 호출자가 평균에서 빼고 수를 센다.
    """
    edge_mse = PIXEL_MAX**2 / np.power(10.0, edge_psnr_y / 10.0)
    flat_mse = PIXEL_MAX**2 / np.power(10.0, flat_psnr_y / 10.0)
    edge_sum = edge_fraction * edge_mse
    total = edge_sum + (1.0 - edge_fraction) * flat_mse
    return (edge_sum / total.where(total > 0)).astype(float)


def compare_diagnoses(model_dir: Path, reference_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """두 진단 출력 폴더를 영상 단위로 비교해 테스트셋별 표를 돌려줍니다.

    각 폴더는 `run_pilot_diagnosis`의 출력(`per_image_diagnosis.csv`, `diagnosis_settings.json`)이어야 한다.
    두 진단의 `edge_top_q`·`shave_border`·`datasets`가 다르거나 (dataset, image_id) 집합이 다르면
    공정한 비교가 아니므로 `ValueError`.

    표의 열(테스트셋마다 한 행, 테스트셋 이름순):
    - `n_images`
    - `delta_<열>`: 영상별 (모델 − 기준)의 평균. 열은 `DELTA_COLUMNS`.
    - `win_rate_<열>`: 모델이 기준보다 나은 영상 비율. PSNR류는 클 때, `mse_cb`·`mse_cr`는 작을 때 이긴 것으로 센다.
    - `model_share_y`, `reference_share_y`: YCbCr 오차 중 Y 비중의 평균.
    - `model_edge_error_share`, `reference_edge_error_share`: Y 오차 중 에지 영역(HR Y Sobel 상위 q%) 비중의 평균.
      Y 오차가 0인 영상은 비중이 정의되지 않아 평균에서 빼고, 뺀 수를 `<side>_edge_share_excluded`에 남긴다.

    함께 돌려주는 설정에는 맞춘 진단 설정과 두 진단의 `commit`이 있다.
    """
    model, model_settings = _load_diagnosis(model_dir)
    reference, reference_settings = _load_diagnosis(reference_dir)
    for key in MATCHED_SETTINGS:
        if key not in model_settings or key not in reference_settings:
            raise ValueError(f"진단 설정에 {key}가 없습니다.")
        if model_settings[key] != reference_settings[key]:
            raise ValueError(
                f"두 진단의 {key}가 다릅니다({model_settings[key]!r} vs {reference_settings[key]!r}). "
                "같은 진단 설정으로 만든 결과만 비교할 수 있습니다."
            )
    model_keys = set(map(tuple, model[_KEYS].to_numpy()))
    reference_keys = set(map(tuple, reference[_KEYS].to_numpy()))
    if model_keys != reference_keys:
        only = sorted(model_keys ^ reference_keys)[:5]
        raise ValueError(f"두 진단의 영상 집합이 다릅니다(한쪽에만 있는 예: {only}).")

    edge_fraction = float(model_settings["edge_top_q"]) / 100.0
    merged = model.merge(reference, on=_KEYS, suffixes=("_model", "_reference"), validate="one_to_one")
    for side in ("model", "reference"):
        merged[f"edge_error_share_{side}"] = _edge_error_share(
            merged[f"edge_psnr_y_{side}"], merged[f"flat_psnr_y_{side}"], edge_fraction
        )

    rows = []
    for dataset, group in merged.groupby("dataset", sort=True):
        row: dict[str, Any] = {"dataset": dataset, "n_images": len(group)}
        for column in DELTA_COLUMNS:
            row[f"delta_{column}"] = float((group[f"{column}_model"] - group[f"{column}_reference"]).mean())
        for column, direction in WIN_COLUMNS.items():
            better = direction * (group[f"{column}_model"] - group[f"{column}_reference"]) > 0
            row[f"win_rate_{column}"] = float(better.mean())
        for side in ("model", "reference"):
            row[f"{side}_share_y"] = float(group[f"share_y_{side}"].mean())
            share = group[f"edge_error_share_{side}"]
            row[f"{side}_edge_error_share"] = float(share.mean(skipna=True))
            row[f"{side}_edge_share_excluded"] = int(share.isna().sum())
        rows.append(row)

    settings = {key: model_settings[key] for key in MATCHED_SETTINGS}
    settings["model_diagnosis_commit"] = model_settings.get("commit")
    settings["reference_diagnosis_commit"] = reference_settings.get("commit")
    return pd.DataFrame(rows), settings
