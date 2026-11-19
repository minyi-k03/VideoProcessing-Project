"""진단 4종 + 최악 영상 목록 생성 (SPEC.md / ADR-0004).

1주차 파일럿 베이스라인 진단과 본 파이프라인 내 진단 분석이 모두 이 함수들을 공유합니다.

공통 규칙:
- 영상 1장 단위: 입력은 (1, C, H, W) float 텐서. 테스트셋 영상 크기가 제각각이라 배치를 쓰지 않는다.
- RGB는 [0, 1], Y는 BT.601 8비트 스케일(16~235). 색 변환은 `common.color`만 쓴다.
- PSNR은 최댓값 255 기준이며, 오차가 0이면 inf.
- 자르기·양자화·가장자리 처리는 하지 않는다. `run_diagnosis_suite`가 8비트 PNG를 읽고
  가장자리를 잘라 넘긴다.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from sr_pipeline.common.color import rgb_to_y, rgb_to_ycbcr

PIXEL_MAX = 255.0
# 주파수 대역 경계 (cycles/pixel). Nyquist 0.5를 3등분한다.
BAND_EDGES = (1.0 / 6.0, 1.0 / 3.0)
RADIAL_BINS = 64
LOG_EPS = 1e-8  # log 스펙트럼에서 크기 0을 피하는 값
WORST_TOP_K = 5

_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))


# ---------------------------------------------------------------- 입력 검증·공용 계산


def _check_pair(sr: torch.Tensor, hr: torch.Tensor, channels: int, name: str) -> None:
    for label, tensor in ((f"sr_{name}", sr), (f"hr_{name}", hr)):
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != channels:
            raise ValueError(f"{label}는 (1, {channels}, H, W) 텐서여야 합니다: {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"{label}는 float 텐서여야 합니다: {tensor.dtype}")
    if sr.shape != hr.shape:
        raise ValueError(f"SR 결과와 HR 영상의 shape가 다릅니다: {tuple(sr.shape)} vs {tuple(hr.shape)}")


def _psnr(mse: float) -> float:
    return math.inf if mse == 0 else 10.0 * math.log10(PIXEL_MAX**2 / mse)


def _sobel_magnitude(y: torch.Tensor) -> torch.Tensor:
    """(1, 1, H, W) → (H, W) Sobel 크기. 가장자리는 복제 패딩."""
    kx = torch.tensor(_SOBEL_X, dtype=y.dtype, device=y.device)
    kernels = torch.stack([kx, kx.T]).unsqueeze(1)
    grad = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kernels)
    return grad.pow(2).sum(dim=1).sqrt()[0]


def _edge_mask(hr_y: torch.Tensor, top_q_percent: float) -> torch.Tensor:
    """HR Y의 Sobel 크기 상위 q% 화소 (H, W) bool 마스크. 개수는 ceil(q% · N)로 정확히 맞춘다."""
    if not 0.0 < top_q_percent < 100.0:
        raise ValueError(f"top_q_percent는 0과 100 사이여야 합니다: {top_q_percent}")
    magnitude = _sobel_magnitude(hr_y).flatten()
    count = math.ceil(magnitude.numel() * top_q_percent / 100.0)
    if count >= magnitude.numel():
        raise ValueError(f"영상이 너무 작아 평탄 영역이 남지 않습니다: {tuple(hr_y.shape)}")
    mask = torch.zeros_like(magnitude, dtype=torch.bool)
    mask[torch.topk(magnitude, count, sorted=False).indices] = True
    return mask.view(hr_y.shape[-2:])


# ---------------------------------------------------------------- 진단 함수


def compute_error_map(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> torch.Tensor:
    """SR 결과와 HR 영상의 화소별 오차 지도 |Y_sr − Y_hr| (H, W)를 계산합니다.

    주 지표 PSNR(Y)와 같은 채널이라 그림과 수치가 맞는다.
    """
    _check_pair(sr_rgb, hr_rgb, 3, "rgb")
    return (rgb_to_y(sr_rgb) - rgb_to_y(hr_rgb)).abs()[0, 0]


def compute_edge_flat_psnr(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
    top_q_percent: float,
) -> dict[str, float]:
    """에지 영역(HR Y Sobel 크기 상위 q%)과 나머지 평탄 영역의 PSNR(Y)을 분리 계산합니다."""
    _check_pair(sr_y, hr_y, 1, "y")
    mask = _edge_mask(hr_y, top_q_percent)
    squared = (sr_y - hr_y)[0, 0].double().pow(2)
    return {
        "edge_psnr_y": _psnr(squared[mask].mean().item()),
        "flat_psnr_y": _psnr(squared[~mask].mean().item()),
    }


def compute_frequency_spectrum_diff(
    sr_y: torch.Tensor,
    hr_y: torch.Tensor,
) -> dict[str, Any]:
    """SR 결과와 HR 영상의 2D FFT 주파수 스펙트럼 및 대역별 에너지를 비교합니다.

    `band_energy_ratio`는 저·중·고 대역에서 SR 에너지 / HR 에너지다. 고주파를 잃으면 high < 1.
    FFT는 화소 수로 나눈다(norm="forward", DC = 평균 밝기). 반지름 프로파일은 0~0.5 cycles/pixel을
    `RADIAL_BINS`개 구간으로 나눠 구간마다 에너지 |F|²를 합한 값이다. 파스발 정리로 전체 합이
    영상의 평균 제곱 밝기와 같으므로, 같은 밝기·무늬면 영상 크기와 관계없이 같은 프로파일이 나와
    크기가 다른 영상끼리 평균해도 큰 영상이 과대 반영되지 않는다.
    """
    _check_pair(sr_y, hr_y, 1, "y")
    height, width = hr_y.shape[-2:]
    freq_y = torch.fft.fftfreq(height, dtype=torch.float64).view(-1, 1)
    freq_x = torch.fft.fftfreq(width, dtype=torch.float64).view(1, -1)
    radius = (freq_y.pow(2) + freq_x.pow(2)).sqrt()

    spectra = {name: torch.fft.fft2(y[0, 0].double(), norm="forward").abs() for name, y in (("sr", sr_y), ("hr", hr_y))}

    bands = {
        "low": radius < BAND_EDGES[0],
        "mid": (radius >= BAND_EDGES[0]) & (radius < BAND_EDGES[1]),
        "high": radius >= BAND_EDGES[1],
    }
    band_energy_ratio = {}
    for band, mask in bands.items():
        sr_energy = spectra["sr"][mask].pow(2).sum().item()
        hr_energy = spectra["hr"][mask].pow(2).sum().item()
        if hr_energy == 0:
            band_energy_ratio[band] = 1.0 if sr_energy == 0 else math.inf
        else:
            band_energy_ratio[band] = sr_energy / hr_energy

    bin_index = torch.clamp((radius / 0.5 * RADIAL_BINS).long(), max=RADIAL_BINS)  # RADIAL_BINS = 0.5 초과(버림)
    radial = {}
    for name, spectrum in spectra.items():
        energy = torch.zeros(RADIAL_BINS + 1, dtype=torch.float64)
        energy.index_add_(0, bin_index.flatten(), spectrum.pow(2).flatten())
        radial[name] = energy[:RADIAL_BINS].numpy()
    radial_freq = (np.arange(RADIAL_BINS) + 0.5) * 0.5 / RADIAL_BINS

    return {
        "sr_log_spectrum": torch.fft.fftshift(torch.log10(spectra["sr"] + LOG_EPS)),
        "hr_log_spectrum": torch.fft.fftshift(torch.log10(spectra["hr"] + LOG_EPS)),
        "radial_freq": radial_freq,
        "sr_radial": radial["sr"],
        "hr_radial": radial["hr"],
        "band_energy_ratio": band_energy_ratio,
    }


def decompose_ycbcr_error(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> dict[str, float]:
    """BT.601 YCbCr 채널별(Y, Cb, Cr) MSE와 전체 오차에서 차지하는 비중을 분해합니다."""
    _check_pair(sr_rgb, hr_rgb, 3, "rgb")
    mse = (rgb_to_ycbcr(sr_rgb) - rgb_to_ycbcr(hr_rgb)).double().pow(2).mean(dim=(0, 2, 3))
    total = mse.sum().item()
    result = {}
    for index, channel in enumerate(("y", "cb", "cr")):
        result[f"mse_{channel}"] = mse[index].item()
        result[f"share_{channel}"] = mse[index].item() / total if total > 0 else 0.0
    return result


def rank_worst_images(per_image_csv: Path, top_k: int = WORST_TOP_K) -> list[dict[str, Any]]:
    """테스트셋별 PSNR(Y)이 가장 낮은 영상 top_k개를 반환합니다. 동점은 image_id 순서."""
    if top_k < 1:
        raise ValueError(f"top_k는 1 이상이어야 합니다: {top_k}")
    table = pd.read_csv(per_image_csv, dtype={"image_id": str})
    missing = {"dataset", "image_id", "psnr_y"} - set(table.columns)
    if missing:
        raise ValueError(f"{per_image_csv}에 필요한 열이 없습니다: {sorted(missing)}")
    if "run_id" in table.columns and table["run_id"].nunique() > 1:
        raise ValueError("여러 실행의 행이 섞여 있습니다. 실행 하나의 per_image 기록만 넣어 주세요.")

    worst = []
    for dataset, group in table.groupby("dataset", sort=True):
        ranked = group.sort_values(["psnr_y", "image_id"], kind="mergesort").head(top_k)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            worst.append({"dataset": dataset, "image_id": row.image_id, "psnr_y": float(row.psnr_y), "rank": rank})
    return worst


# ---------------------------------------------------------------- 전체 실행


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _shave(image: torch.Tensor, border: int) -> torch.Tensor:
    return image[..., border:-border, border:-border] if border > 0 else image


def _collect_pairs(sr_dir: Path, hr_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """(dataset, image_id, sr_path, hr_path) 목록. HR 구조는 T1.4 `prepare_testsets` 산출물과 같다."""
    datasets = sorted(path.name for path in hr_dir.iterdir() if (path / "HR").is_dir())
    if not datasets:
        raise ValueError(f"{hr_dir} 아래에 <데이터셋>/HR 폴더가 없습니다.")
    pairs = []
    for dataset in datasets:
        for hr_path in sorted((hr_dir / dataset / "HR").glob("*.png")):
            sr_path = sr_dir / dataset / hr_path.name
            if not sr_path.is_file():
                raise ValueError(f"HR 영상에 대응하는 SR 결과가 없습니다: {sr_path}")
            pairs.append((dataset, hr_path.stem, sr_path, hr_path))
    return pairs


def _save_figures(
    output_dir: Path,
    per_image: pd.DataFrame,
    worst: list[dict[str, Any]],
    error_maps: dict[tuple[str, str], np.ndarray],
    radial: dict[str, dict[str, list[np.ndarray]]],
    radial_freq: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error_dir = output_dir / "error_maps"
    error_dir.mkdir(exist_ok=True)
    for item in worst:
        key = (item["dataset"], item["image_id"])
        fig, ax = plt.subplots(figsize=(6, 5))
        shown = ax.imshow(error_maps[key], cmap="inferno")
        fig.colorbar(shown, ax=ax, label="|Y_SR − Y_HR| (8-bit)")
        ax.set_title(f"{key[0]} {key[1]}  PSNR(Y) {item['psnr_y']:.2f} dB")
        ax.axis("off")
        fig.savefig(error_dir / f"{key[0]}_{key[1]}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    for dataset, profiles in radial.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(radial_freq, np.mean(profiles["hr"], axis=0), label="HR")
        ax.semilogy(radial_freq, np.mean(profiles["sr"], axis=0), label="SR")
        for edge in BAND_EDGES:
            ax.axvline(edge, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("spatial frequency (cycles/pixel)")
        ax.set_ylabel("energy per band |F|² (Y)")
        ax.set_title(f"{dataset}: radial spectrum")
        ax.legend()
        fig.savefig(output_dir / f"spectrum_{dataset}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    finite = per_image.replace([np.inf, -np.inf], np.nan)
    means = finite.groupby("dataset")[["edge_psnr_y", "flat_psnr_y"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    means.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("PSNR(Y) (dB)")
    ax.set_title("edge vs flat PSNR(Y)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(output_dir / "edge_flat_psnr.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    shares = per_image.groupby("dataset")[["share_y", "share_cb", "share_cr"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    shares.plot.bar(ax=ax, stacked=True, rot=0)
    ax.set_ylabel("share of YCbCr MSE")
    ax.set_ylim(0, 1)
    ax.set_title("YCbCr error share")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(output_dir / "ycbcr_error_share.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_diagnosis_suite(
    sr_dir: Path,
    hr_dir: Path,
    output_dir: Path,
    *,
    top_q_percent: float,
    shave_border: int,
) -> None:
    """진단 4종 그림/수치와 테스트셋별 최악 영상 목록을 `output_dir`에 저장합니다.

    입력: `hr_dir/<데이터셋>/HR/<image_id>.png`, `sr_dir/<데이터셋>/<image_id>.png` (8비트 PNG).
    `top_q_percent`(`edge_top_q`)와 `shave_border`는 `fair_comparison.yaml`에서 읽어 호출하는 쪽이
    넘긴다. 기본값을 두지 않는 이유: 공정 비교 조건 값은 그 파일 한 곳에만 있고, `edge_top_q`는
    T4.2에서 확정되기 전이라 코드에 임의 값을 박으면 확정값과 다른 진단이 나올 수 있다.
    """
    sr_dir, hr_dir, output_dir = Path(sr_dir), Path(hr_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    error_maps: dict[tuple[str, str], np.ndarray] = {}
    radial: dict[str, dict[str, list[np.ndarray]]] = {}
    radial_freq = np.empty(0)
    for dataset, image_id, sr_path, hr_path in _collect_pairs(sr_dir, hr_dir):
        sr_rgb, hr_rgb = _load_rgb(sr_path), _load_rgb(hr_path)
        if sr_rgb.shape != hr_rgb.shape:
            raise ValueError(f"SR 결과와 HR 영상의 크기가 다릅니다: {sr_path} {tuple(sr_rgb.shape)} vs {tuple(hr_rgb.shape)}")
        sr_rgb, hr_rgb = _shave(sr_rgb, shave_border), _shave(hr_rgb, shave_border)
        sr_y, hr_y = rgb_to_y(sr_rgb), rgb_to_y(hr_rgb)

        spectrum = compute_frequency_spectrum_diff(sr_y, hr_y)
        radial_freq = spectrum["radial_freq"]
        profiles = radial.setdefault(dataset, {"sr": [], "hr": []})
        profiles["sr"].append(spectrum["sr_radial"])
        profiles["hr"].append(spectrum["hr_radial"])
        error_maps[(dataset, image_id)] = compute_error_map(sr_rgb, hr_rgb).numpy()

        rows.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "psnr_y": _psnr((sr_y - hr_y).double().pow(2).mean().item()),
                **compute_edge_flat_psnr(sr_y, hr_y, top_q_percent),
                **decompose_ycbcr_error(sr_rgb, hr_rgb),
                **{f"band_ratio_{band}": value for band, value in spectrum["band_energy_ratio"].items()},
            }
        )

    per_image = pd.DataFrame(rows)
    per_image.to_csv(output_dir / "per_image_diagnosis.csv", index=False)
    per_image.drop(columns="image_id").groupby("dataset").mean().to_csv(output_dir / "summary_diagnosis.csv")
    worst = rank_worst_images(output_dir / "per_image_diagnosis.csv")
    pd.DataFrame(worst).to_csv(output_dir / "worst_images.csv", index=False)
    _save_figures(output_dir, per_image, worst, error_maps, radial, radial_freq)


# ---------------------------------------------------------------- 파일럿 진단 (T2.5)

DIAGNOSIS_SETTINGS_FILENAME = "diagnosis_settings.json"
# 진단이 만들고 다시 실행할 때 통째로 바꾸는 산출물(glob 패턴). 같은 폴더의 다른 파일
# (예: 파일럿 학습 설정 기록, .gitkeep)은 건드리지 않는다.
DIAGNOSIS_OUTPUT_PATTERNS = (
    "per_image_diagnosis.csv",
    "summary_diagnosis.csv",
    "worst_images.csv",
    "edge_flat_psnr.png",
    "ycbcr_error_share.png",
    "spectrum_*.png",
    "error_maps",
    DIAGNOSIS_SETTINGS_FILENAME,
)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}은(는) 키-값 묶음이어야 합니다: {type(value).__name__}")
    return value


def _require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name}은(는) 숫자여야 합니다: {value!r}")
    return float(value)


def resolve_diagnosis_settings(fair_comparison: dict[str, Any], edge_top_q: float | None) -> dict[str, Any]:
    """`fair_comparison.yaml` 내용과 호출자가 넘긴 `edge_top_q`로 진단에 쓸 값을 정합니다.

    - `shave_border`·`datasets`는 `fixed_rules`에서만 읽는다(호출자가 바꿀 수 없다).
    - `pilot_determined.edge_top_q`가 있으면 그 값을 쓰고, 넘긴 `edge_top_q`가 다르면 오류.
    - 아직 `null`(T4.2 확정 전)이면 `edge_top_q`를 넘겨야 한다. 코드에 임의 기본값을 두지 않고,
      파일럿에서 쓴 값과 출처를 결과에 남기기 위해서다.
    """
    fair_comparison = _require_mapping(fair_comparison, "fair_comparison.yaml")
    fixed_rules = _require_mapping(fair_comparison.get("fixed_rules"), "fixed_rules")
    pilot_determined = _require_mapping(fair_comparison.get("pilot_determined", {}), "pilot_determined")

    shave_border = fixed_rules.get("shave_border")
    if not isinstance(shave_border, int) or isinstance(shave_border, bool) or shave_border < 0:
        raise ValueError(f"fixed_rules.shave_border는 0 이상의 정수여야 합니다: {shave_border!r}")

    datasets = fixed_rules.get("datasets")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(name, str) and name for name in datasets)
        or len(set(datasets)) != len(datasets)
    ):
        raise ValueError(f"fixed_rules.datasets는 중복 없는 테스트셋 이름 목록이어야 합니다: {datasets!r}")

    configured_q = pilot_determined.get("edge_top_q")
    if configured_q is None:
        if edge_top_q is None:
            raise ValueError(
                "fair_comparison.yaml의 pilot_determined.edge_top_q가 아직 null입니다(T4.2 확정 전). "
                "파일럿에서 쓸 edge_top_q 값을 넘기세요."
            )
        top_q, source = _require_number(edge_top_q, "edge_top_q"), "command_line"
    else:
        configured_q = _require_number(configured_q, "pilot_determined.edge_top_q")
        if edge_top_q is not None and _require_number(edge_top_q, "edge_top_q") != configured_q:
            raise ValueError(
                f"edge_top_q {edge_top_q}가 fair_comparison.yaml의 edge_top_q {configured_q}와 다릅니다. "
                "확정된 공정 비교 조건 값은 호출할 때 바꿀 수 없습니다."
            )
        top_q, source = configured_q, "fair_comparison"
    if not 0.0 < top_q < 100.0:
        raise ValueError(f"edge_top_q는 0과 100 사이여야 합니다: {top_q}")
    return {"edge_top_q": top_q, "edge_top_q_source": source, "shave_border": shave_border, "datasets": list(datasets)}


def _check_hr_datasets(hr_dir: Path, datasets: list[str]) -> None:
    """HR 영상 폴더의 `<데이터셋>/HR` 목록이 공정 비교 조건의 테스트셋과 정확히 같은지 확인합니다."""
    found = {path.name for path in Path(hr_dir).iterdir() if (path / "HR").is_dir()} if Path(hr_dir).is_dir() else set()
    missing, extra = sorted(set(datasets) - found), sorted(found - set(datasets))
    if missing or extra:
        raise ValueError(
            f"HR 영상 폴더의 테스트셋이 fixed_rules.datasets와 다릅니다. 없음: {missing}, 목록에 없는 폴더: {extra}"
        )


def _replace_diagnosis_outputs(staging_dir: Path, output_dir: Path) -> None:
    """`output_dir`의 진단 산출물을 지우고 `staging_dir`에 만든 새 산출물로 바꿉니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in DIAGNOSIS_OUTPUT_PATTERNS:
        for stale in output_dir.glob(pattern):
            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    for fresh in staging_dir.iterdir():
        shutil.move(str(fresh), str(output_dir / fresh.name))


def run_pilot_diagnosis(
    sr_dir: Path,
    hr_dir: Path,
    output_dir: Path,
    fair_comparison: dict[str, Any],
    *,
    commit: str,
    edge_top_q: float | None = None,
) -> dict[str, Any]:
    """파일럿 SR 결과로 진단 4종을 돌리고, 쓴 설정을 `diagnosis_settings.json`에 남깁니다 (US 51).

    입력 폴더 구조와 산출물은 `run_diagnosis_suite`와 같다. HR 영상 폴더의 테스트셋은
    `fixed_rules.datasets`와 정확히 같아야 한다. `commit`은 이 진단을 실행한 코드 커밋이다.

    산출물은 옆의 임시 폴더에 모두 만든 뒤 `output_dir`의 이전 진단 산출물과 통째로 바꾼다.
    그래서 다시 실행해도 이전 오차 지도가 섞이지 않고, 도중에 실패하면 `output_dir`은 그대로다.
    진단이 만들지 않은 파일(파일럿 학습 설정 기록, `.gitkeep` 등)은 남는다.
    결과가 Public 저장소에 커밋되므로 입력 경로는 기록하지 않는다. 반환값은 기록한 설정이다.
    """
    settings = resolve_diagnosis_settings(fair_comparison, edge_top_q)
    if not isinstance(commit, str) or not commit.strip():
        raise ValueError(f"commit은 이 진단을 실행한 코드 커밋이어야 합니다: {commit!r}")
    _check_hr_datasets(Path(hr_dir), settings["datasets"])
    settings["commit"] = commit.strip()

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        run_diagnosis_suite(
            sr_dir,
            hr_dir,
            staging_dir,
            top_q_percent=settings["edge_top_q"],
            shave_border=settings["shave_border"],
        )
        with (staging_dir / DIAGNOSIS_SETTINGS_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(settings, file, ensure_ascii=False, indent=2)
        _replace_diagnosis_outputs(staging_dir, output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return settings
