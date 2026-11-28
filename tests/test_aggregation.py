"""자동 집계 보조 경계 테스트 (SPEC.md Testing Decisions §3).

입력은 손으로 계산한 값을 넣은 고정된 가짜 실행 기록이다. 기대값은 코드가 아니라 아래 주석의 손계산에서 온다.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from sr_pipeline.reporting.aggregate import (
    build_ablation_table,
    build_b_pre_rgb_diagnostics_table,
    build_progress_table,
    build_training_time_table,
    compute_per_image_deltas,
    compute_result_tables,
    export_report_assets,
    format_result_report_table,
    load_run_records,
    select_case_study_images,
    validate_unique_runs,
)

POOL = "pool_x"
SEEDS = (0, 1, 2)
METRICS = ("psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips")
SUMMARY_COLUMNS = [
    "run_id", "commit", "config_id", "seed", "patch_pool_id", "switches", "dataset",
    "psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "lpips", "psnr_y_pre_rgb",
    "params", "infer_ms_net", "infer_ms_total", "train_minutes", "status",
]  # fmt: skip
PER_IMAGE_COLUMNS = ["run_id", "dataset", "image_id", *METRICS]


def _row(config_id, seed, psnr_y, *, dataset="Set5", pool=POOL, status="complete", commit="c0") -> dict:
    return {
        "run_id": f"{commit}-{config_id}-s{seed}",
        "commit": commit,
        "config_id": config_id,
        "seed": seed,
        "patch_pool_id": pool,
        "switches": "none" if config_id in ("baseline", "bicubic") else config_id,
        "dataset": dataset,
        "psnr_y": psnr_y,
        "ssim_y": 0.9,
        "psnr_rgb": 28.0,
        "edge_psnr": 25.0,
        "lpips": 0.1,
        "psnr_y_pre_rgb": None,
        "params": 100,
        "infer_ms_net": 1.0,
        "infer_ms_total": 2.0,
        "train_minutes": 30.0,
        "status": status,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _fake_records() -> pd.DataFrame:
    """Set5 psnr_y. baseline [30.0, 30.2, 29.8], A [30.5, 30.6, 30.1], B [30.5, 29.9, 30.0]."""
    values = {
        "baseline": (30.0, 30.2, 29.8),
        "A": (30.5, 30.6, 30.1),
        "B": (30.5, 29.9, 30.0),
    }
    return _frame([_row(cfg, s, v) for cfg, vs in values.items() for s, v in zip(SEEDS, vs)])


def _result(tables: dict, config_id: str, metric: str = "psnr_y", dataset: str = "Set5") -> pd.Series:
    df = tables["results"]
    hit = df[(df.config_id == config_id) & (df.metric == metric) & (df.dataset == dataset)]
    assert len(hit) == 1
    return hit.iloc[0]


class TestMeanStdDelta:
    def test_mean_std_delta_and_seed_delta_for_consistent_improvement(self) -> None:
        tables = compute_result_tables(_fake_records(), expected_seeds=SEEDS)
        base = _result(tables, "baseline")
        a = _result(tables, "A")

        # 손계산: baseline 평균 30.0, A 평균 30.4. A의 표준편차(ddof=1)는 sqrt(0.14/2).
        assert base["mean"] == pytest.approx(30.0)
        assert base["std"] == pytest.approx(math.sqrt(0.08 / 2))  # 편차 0, 0.2, -0.2
        assert a["mean"] == pytest.approx(30.4)
        assert a["std"] == pytest.approx(math.sqrt(0.14 / 2))
        assert a["delta"] == pytest.approx(0.4)
        # 시드별 Δ = [0.5, 0.4, 0.3] -> 부호 모두 양수, 표준편차 0.1 < 평균 Δ 0.4
        assert a["seed_delta_std"] == pytest.approx(0.1)
        assert a["delta_sign"] == "positive"
        assert bool(a["delta_sign_consistent"]) is True
        assert bool(a["delta_exceeds_seed_delta_std"]) is True

        seed_deltas = tables["seed_deltas"]
        got = seed_deltas[(seed_deltas.config_id == "A") & (seed_deltas.metric == "psnr_y")].sort_values("seed")
        assert list(got["delta"]) == pytest.approx([0.5, 0.4, 0.3])

    def test_mixed_sign_seed_deltas_are_flagged(self) -> None:
        tables = compute_result_tables(_fake_records(), expected_seeds=SEEDS)
        b = _result(tables, "B")

        # 시드별 Δ = [0.5, -0.3, 0.2] -> 평균 Δ 0.1333, 표준편차 sqrt(0.3267/2)=0.4041
        assert b["delta"] == pytest.approx(0.4 / 3)
        assert b["seed_delta_std"] == pytest.approx(math.sqrt(0.326667 / 2), abs=1e-4)
        assert b["delta_sign"] == "mixed"
        assert bool(b["delta_sign_consistent"]) is False
        assert bool(b["delta_exceeds_seed_delta_std"]) is False

    def test_zero_seed_delta_is_not_a_consistent_direction(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 30.0), _row("A", 1, 30.1), _row("A", 2, 30.2)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert a["delta_sign"] == "mixed"
        assert bool(a["delta_sign_consistent"]) is False

    def test_baseline_has_no_delta_and_datasets_are_separate(self) -> None:
        rows = [_row("baseline", s, 30.0 + s) for s in SEEDS] + [_row("baseline", s, 20.0, dataset="Set14") for s in SEEDS]
        rows += [_row("A", s, 31.0 + s) for s in SEEDS] + [_row("A", s, 21.0, dataset="Set14") for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        assert math.isnan(_result(tables, "baseline")["delta"])
        assert _result(tables, "A", dataset="Set5")["delta"] == pytest.approx(1.0)
        assert _result(tables, "A", dataset="Set14")["delta"] == pytest.approx(1.0)
        assert _result(tables, "A", dataset="Set14")["mean"] == pytest.approx(21.0)

    def test_all_negative_seed_deltas_are_consistent_but_signed_mean_does_not_exceed_std(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 29.5), _row("A", 1, 29.8), _row("A", 2, 29.9)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        # 시드별 Δ = [-0.5, -0.2, -0.1] -> 부호 모두 음수(consistent=True), 부호 있는 평균 -0.2667 <= std 0.2082 (False)
        assert a["delta"] == pytest.approx(-0.8 / 3)
        assert a["delta_sign"] == "negative"
        assert bool(a["delta_sign_consistent"]) is True
        assert bool(a["delta_exceeds_seed_delta_std"]) is False

    def test_uniform_negative_seed_deltas_yield_false_for_signed_mean_vs_std(self) -> None:
        # 세 시드 PSNR(Y) Δ가 모두 -1이면 평균 -1, 표준편차 0 -> 부호 있는 평균 Δ > std는 False (SPEC US59·ADR-0003)
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", s, 29.0) for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert a["delta"] == pytest.approx(-1.0)
        assert a["seed_delta_std"] == pytest.approx(0.0)
        assert a["delta_sign"] == "negative"
        assert bool(a["delta_sign_consistent"]) is True
        assert bool(a["delta_exceeds_seed_delta_std"]) is False

    def test_positive_inf_psnr_marks_undefined_std_and_delta_as_unjudgable(self) -> None:
        # 평탄 합성 영상 등에서 PSNR이 정상적으로 +inf인 경우:
        # (1) 구성과 베이스라인이 모두 +inf -> delta=NaN, std=NaN, 부호·판정 모두 판정 불가(<NA>)
        # (2) 구성만 +inf, 베이스라인은 유한 -> mean=+inf, delta=+inf, std=NaN, 부호는 positive이나 평균 Δ vs std 판정은 <NA>
        rows = [_row("baseline", s, math.inf) for s in SEEDS] + [_row("A", s, math.inf) for s in SEEDS]
        rows += [_row("baseline", s, 30.0, dataset="Set14") for s in SEEDS]
        rows += [_row("A", s, math.inf, dataset="Set14") for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)

        both_inf = _result(tables, "A", dataset="Set5")
        assert math.isinf(both_inf["mean"]) and both_inf["mean"] > 0
        assert math.isnan(both_inf["std"])
        assert math.isnan(both_inf["delta"])
        assert math.isnan(both_inf["seed_delta_std"])
        assert both_inf["delta_sign"] == ""
        assert both_inf["delta_sign_consistent"] is pd.NA
        assert both_inf["delta_exceeds_seed_delta_std"] is pd.NA

        config_inf = _result(tables, "A", dataset="Set14")
        assert math.isinf(config_inf["mean"]) and config_inf["mean"] > 0
        assert math.isnan(config_inf["std"])
        assert math.isinf(config_inf["delta"]) and config_inf["delta"] > 0
        assert math.isnan(config_inf["seed_delta_std"])
        assert config_inf["delta_sign"] == "positive"
        assert bool(config_inf["delta_sign_consistent"]) is True
        assert config_inf["delta_exceeds_seed_delta_std"] is pd.NA

    def test_single_seed_pair_gives_no_direction_judgement(self) -> None:
        rows = [_row("baseline", 0, 30.0), _row("A", 0, 31.0)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert a["delta"] == pytest.approx(1.0)
        assert a["delta_sign"] == "positive"  # 짝지은 시드별 Δ를 그대로 묘사할 뿐 판정이 아니다
        assert math.isnan(a["std"]) and math.isnan(a["seed_delta_std"])
        assert a["delta_sign_consistent"] is pd.NA
        assert a["delta_exceeds_seed_delta_std"] is pd.NA

    def test_direction_judgement_needs_every_expected_seed_paired(self) -> None:
        # 시드 2가 빠진 A: 시드별 Δ [1.0, 1.2] 둘은 있지만 "3개 모두"를 판정할 수 없다.
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 31.0), _row("A", 1, 31.2)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert a["seed_delta_std"] == pytest.approx(math.sqrt(0.02))
        assert a["delta_sign_consistent"] is pd.NA
        assert a["delta_exceeds_seed_delta_std"] is pd.NA

    def test_numbers_trace_back_to_run_ids(self) -> None:
        tables = compute_result_tables(_fake_records(), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert a["run_ids"] == "c0-A-s0,c0-A-s1,c0-A-s2"
        assert a["baseline_run_ids"] == "c0-baseline-s0,c0-baseline-s1,c0-baseline-s2"
        assert _result(tables, "baseline")["baseline_run_ids"] == ""

        seed_deltas = tables["seed_deltas"]
        got = seed_deltas[(seed_deltas.config_id == "A") & (seed_deltas.metric == "psnr_y")].sort_values("seed")
        assert list(got["run_id"]) == ["c0-A-s0", "c0-A-s1", "c0-A-s2"]
        assert list(got["baseline_run_id"]) == ["c0-baseline-s0", "c0-baseline-s1", "c0-baseline-s2"]

    def test_bicubic_delta_traces_to_baseline_runs(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("bicubic", 0, 28.0)]
        bic = _result(compute_result_tables(_frame(rows), expected_seeds=SEEDS), "bicubic")
        assert bic["run_ids"] == "c0-bicubic-s0"
        assert bic["baseline_run_ids"] == "c0-baseline-s0,c0-baseline-s1,c0-baseline-s2"

    def test_every_metric_is_aggregated(self) -> None:
        tables = compute_result_tables(_fake_records(), expected_seeds=SEEDS)
        assert set(tables["results"].metric) == set(METRICS)

    def test_missing_baseline_leaves_delta_undefined(self) -> None:
        rows = [_row("A", s, 30.0) for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        assert math.isnan(_result(tables, "A")["delta"])

    def test_delta_is_only_taken_against_baseline_of_same_patch_pool(self) -> None:
        rows = [_row("baseline", s, 30.0, pool="p1") for s in SEEDS]
        rows += [_row("A", s, 31.0, pool="p1") for s in SEEDS]
        rows += [_row("A", s, 40.0, pool="p2") for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        df = tables["results"]
        p1 = df[(df.config_id == "A") & (df.metric == "psnr_y") & (df.patch_pool_id == "p1")].iloc[0]
        p2 = df[(df.config_id == "A") & (df.metric == "psnr_y") & (df.patch_pool_id == "p2")].iloc[0]
        assert p1["delta"] == pytest.approx(1.0)
        assert math.isnan(p2["delta"])


class TestCompletenessAndDuplicates:
    def test_missing_seed_is_marked_incomplete(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 31.0), _row("A", 2, 31.0)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert bool(a["is_complete"]) is False
        assert a["missing_seeds"] == "1"
        assert a["incomplete_seeds"] == ""
        assert a["n_seeds"] == 2
        assert bool(_result(tables, "baseline")["is_complete"]) is True

    def test_incomplete_status_run_is_excluded_from_metrics_and_tracked_separately(self) -> None:
        # 완료 시드 둘과 베이스라인이 30이고 미완료 시드 하나가 50이면, 평균·Δ·시드 짝짓기는 완료 시드만 쓴다(SPEC US20).
        rows = [_row("baseline", s, 30.0) for s in SEEDS]
        rows += [_row("A", 0, 30.0), _row("A", 1, 50.0, status="incomplete"), _row("A", 2, 30.0)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert bool(a["is_complete"]) is False
        assert a["missing_seeds"] == ""
        assert a["incomplete_seeds"] == "1"
        assert a["n_seeds"] == 2
        assert a["seeds"] == "0,2"
        assert a["mean"] == pytest.approx(30.0)
        assert a["std"] == pytest.approx(0.0)
        assert a["delta"] == pytest.approx(0.0)
        assert a["run_ids"] == "c0-A-s0,c0-A-s2"
        assert a["incomplete_run_ids"] == "c0-A-s1"
        assert a["baseline_run_ids"] == "c0-baseline-s0,c0-baseline-s1,c0-baseline-s2"
        assert a["delta_sign_consistent"] is pd.NA
        assert a["delta_exceeds_seed_delta_std"] is pd.NA

        paired_seeds = sorted(
            tables["seed_deltas"][(tables["seed_deltas"].config_id == "A") & (tables["seed_deltas"].metric == "psnr_y")][
                "seed"
            ]
        )
        assert paired_seeds == [0, 2]

    def test_incomplete_baseline_run_is_excluded_from_delta_and_pairing(self) -> None:
        rows = [_row("baseline", 0, 30.0), _row("baseline", 1, 50.0, status="incomplete"), _row("baseline", 2, 30.0)]
        rows += [_row("A", s, 31.0) for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        base = _result(tables, "baseline")
        a = _result(tables, "A")
        assert bool(base["is_complete"]) is False
        assert base["mean"] == pytest.approx(30.0)
        assert base["run_ids"] == "c0-baseline-s0,c0-baseline-s2"
        assert base["incomplete_run_ids"] == "c0-baseline-s1"
        assert bool(a["baseline_is_complete"]) is False
        assert a["delta"] == pytest.approx(1.0)
        assert a["baseline_run_ids"] == "c0-baseline-s0,c0-baseline-s2"
        assert a["delta_sign_consistent"] is pd.NA
        assert a["delta_exceeds_seed_delta_std"] is pd.NA

    def test_config_with_only_incomplete_runs_has_no_metrics_or_delta(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 50.0, status="incomplete")]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert bool(a["is_complete"]) is False
        assert a["n_seeds"] == 0
        assert a["seeds"] == ""
        assert math.isnan(a["mean"]) and math.isnan(a["std"]) and math.isnan(a["delta"])
        assert a["run_ids"] == ""
        assert a["incomplete_run_ids"] == "c0-A-s0"
        assert a["baseline_run_ids"] == ""
        assert a["delta_sign_consistent"] is pd.NA
        assert a["delta_exceeds_seed_delta_std"] is pd.NA

    def test_incomplete_baseline_is_flagged_on_compared_configs(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in (0, 1)] + [_row("A", s, 31.0) for s in SEEDS]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        assert bool(_result(tables, "A")["baseline_is_complete"]) is False
        assert bool(_result(tables, "A")["is_complete"]) is True

    def test_bicubic_needs_no_three_seeds(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("bicubic", 0, 28.0)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        bic = _result(tables, "bicubic")
        assert bool(bic["is_complete"]) is True
        assert bic["delta"] == pytest.approx(-2.0)
        assert math.isnan(bic["seed_delta_std"])
        assert bic["delta_sign"] == ""

    def test_bicubic_with_incomplete_status_is_marked_incomplete(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("bicubic", 0, 28.0, status="incomplete")]
        bic = _result(compute_result_tables(_frame(rows), expected_seeds=SEEDS), "bicubic")
        assert bool(bic["is_complete"]) is False
        assert bic["incomplete_seeds"] == "0"
        assert bic["missing_seeds"] == ""
        assert bic["n_seeds"] == 0
        assert math.isnan(bic["mean"]) and math.isnan(bic["delta"])
        assert bic["run_ids"] == ""
        assert bic["incomplete_run_ids"] == "c0-bicubic-s0"
        assert bic["baseline_run_ids"] == ""

    def test_empty_expected_seeds_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected_seeds"):
            compute_result_tables(_fake_records(), expected_seeds=())

    def test_seed_outside_expected_seeds_is_rejected(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 7, 31.0)]
        with pytest.raises(ValueError, match="시드"):
            compute_result_tables(_frame(rows), expected_seeds=SEEDS)

    def test_duplicate_config_seed_pool_is_an_error_even_across_commits(self) -> None:
        rows = [_row("A", 0, 31.0, commit="c0"), _row("A", 0, 31.5, commit="c1")]
        with pytest.raises(ValueError, match="중복") as err:
            validate_unique_runs(_frame(rows))
        assert "c0-A-s0" in str(err.value) and "c1-A-s0" in str(err.value)

    def test_duplicate_rows_for_same_dataset_is_an_error(self) -> None:
        rows = [_row("A", 0, 31.0), _row("A", 0, 31.0)]
        with pytest.raises(ValueError, match="중복"):
            validate_unique_runs(_frame(rows))

    def test_same_config_and_seed_in_different_patch_pools_is_not_a_duplicate(self) -> None:
        rows = [_row("A", 0, 31.0, pool="p1"), _row("A", 0, 31.0, pool="p2")]
        validate_unique_runs(_frame(rows))

    def test_one_run_with_several_datasets_is_not_a_duplicate(self) -> None:
        rows = [_row("A", 0, 31.0, dataset=d) for d in ("Set5", "Set14", "BSD100", "Urban100")]
        validate_unique_runs(_frame(rows))

    def test_compute_refuses_duplicates(self) -> None:
        rows = [_row("A", 0, 31.0, commit="c0"), _row("A", 0, 31.0, commit="c1")]
        with pytest.raises(ValueError, match="중복"):
            compute_result_tables(_frame(rows), expected_seeds=SEEDS)


class TestDirectFrameInput:
    """CSV를 거치지 않고 DataFrame을 직접 넘겨도 결측 키가 groupby에서 조용히 사라지지 않아야 한다."""

    @pytest.mark.parametrize("column", ["run_id", "config_id", "seed", "patch_pool_id", "dataset"])
    def test_validate_unique_runs_rejects_missing_key(self, column: str) -> None:
        rows = [_row("A", 0, 31.0, commit="c0"), _row("A", 0, 31.5, commit="c1")]
        frame = _frame(rows).astype({column: object})
        frame[column] = None
        with pytest.raises(ValueError, match=column):
            validate_unique_runs(frame)

    def test_compute_rejects_missing_patch_pool(self) -> None:
        rows = [_row("baseline", s, 30.0, pool=None) for s in SEEDS]
        with pytest.raises(ValueError, match="patch_pool_id"):
            compute_result_tables(_frame(rows), expected_seeds=SEEDS)

    @pytest.mark.parametrize(("column", "value"), [("lpips", math.nan), ("lpips", math.inf), ("psnr_y", -math.inf)])
    def test_compute_rejects_disallowed_non_finite_metric(self, column: str, value: float) -> None:
        frame = _fake_records()
        frame.loc[0, column] = value
        with pytest.raises(ValueError, match=column):
            compute_result_tables(frame, expected_seeds=SEEDS)

    def test_compute_rejects_unknown_status(self) -> None:
        frame = _fake_records()
        frame.loc[0, "status"] = "done"
        with pytest.raises(ValueError, match="status"):
            compute_result_tables(frame, expected_seeds=SEEDS)


class TestLoadRunRecords:
    def _write_run(self, root: Path, row: dict, images: list[tuple[str, float]]) -> None:
        run_dir = root / row["run_id"]
        run_dir.mkdir(parents=True)
        pd.DataFrame([row], columns=SUMMARY_COLUMNS).to_csv(run_dir / "summary.csv", index=False)
        per_image = [
            {"run_id": row["run_id"], "dataset": row["dataset"], "image_id": i, "psnr_y": v, "ssim_y": 0.9,
             "psnr_rgb": 28.0, "edge_psnr": 25.0, "lpips": 0.1}
            for i, v in images
        ]  # fmt: skip
        pd.DataFrame(per_image, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)

    def test_loads_all_runs_and_keeps_switches_none_and_blank_pre_rgb(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0), ("im2", 31.0)])
        self._write_run(tmp_path, _row("A", 0, 31.0), [("im1", 31.0), ("im2", 32.0)])
        summary, per_image = load_run_records(tmp_path)

        assert len(summary) == 2 and len(per_image) == 4
        assert set(summary.switches) == {"none", "A"}
        assert summary.psnr_y_pre_rgb.isna().all()
        assert set(per_image.run_id) == set(summary.run_id)

    def test_rejects_empty_runs_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="실행 기록"):
            load_run_records(tmp_path)

    def test_rejects_missing_column(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "summary.csv"
        pd.read_csv(path).drop(columns=["lpips"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="lpips"):
            load_run_records(tmp_path)

    def test_rejects_summary_without_per_image(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        (tmp_path / "c0-baseline-s0" / "per_image.csv").unlink()
        with pytest.raises(ValueError, match="per_image"):
            load_run_records(tmp_path)

    def test_rejects_per_image_of_unknown_run(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "per_image.csv"
        pd.read_csv(path).assign(run_id="c9-ghost-s0").to_csv(path, index=False)
        with pytest.raises(ValueError, match="c9-ghost-s0"):
            load_run_records(tmp_path)

    def test_rejects_per_image_without_summary(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        self._write_run(tmp_path, _row("A", 0, 31.0), [("im1", 31.0)])
        (tmp_path / "c0-A-s0" / "summary.csv").unlink()
        with pytest.raises(ValueError, match="summary"):
            load_run_records(tmp_path)

    @pytest.mark.parametrize("filename", ["summary.csv", "per_image.csv"])
    def test_rejects_header_only_csv(self, tmp_path: Path, filename: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / filename
        pd.read_csv(path).iloc[0:0].to_csv(path, index=False)
        with pytest.raises(ValueError, match="행이 없습니다"):
            load_run_records(tmp_path)

    @pytest.mark.parametrize(
        "column",
        ["patch_pool_id", "dataset", "psnr_y", "commit", "switches", "params",
         "infer_ms_net", "infer_ms_total", "train_minutes"],
    )  # fmt: skip
    def test_rejects_blank_required_value(self, tmp_path: Path, column: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "summary.csv"
        pd.read_csv(path).assign(**{column: None}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            load_run_records(tmp_path)

    @pytest.mark.parametrize("column", ["patch_pool_id", "commit", "config_id"])
    def test_rejects_whitespace_only_string(self, tmp_path: Path, column: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "summary.csv"
        pd.read_csv(path).assign(**{column: "   "}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            load_run_records(tmp_path)

    @pytest.mark.parametrize(
        ("column", "value"),
        [("psnr_y", "NaN"), ("psnr_y", "-inf"), ("lpips", "nan"), ("ssim_y", "inf"), ("params", "abc"), ("psnr_y_pre_rgb", "-inf")],
    )
    def test_rejects_disallowed_non_finite_or_non_numeric_summary_value(self, tmp_path: Path, column: str, value: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "summary.csv"
        pd.read_csv(path).assign(**{column: value}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            load_run_records(tmp_path)

    @pytest.mark.parametrize(("column", "value"), [("ssim_y", "inf"), ("lpips", "inf"), ("edge_psnr", "-inf"), ("psnr_y", "NaN")])
    def test_rejects_disallowed_non_finite_per_image_value(self, tmp_path: Path, column: str, value: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "per_image.csv"
        pd.read_csv(path).assign(**{column: value}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            load_run_records(tmp_path)

    def test_loads_and_aggregates_flat_synthetic_bicubic_csv_written_by_evaluator(self, tmp_path: Path) -> None:
        """평가기가 평탄 합성 영상으로 쓴 Bicubic 하한 CSV(+inf PSNR)가 집계까지 정상 통과해야 한다."""
        import numpy as np
        import torch
        import yaml
        from PIL import Image

        from sr_pipeline.data.prepare import prepare_testsets
        from sr_pipeline.evaluation.evaluator import evaluate_model

        class _StubLpips(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return (x - y).pow(2).mean().view(1, 1, 1, 1)

        repo_root = Path(__file__).resolve().parents[1]
        fair = yaml.safe_load((repo_root / "configs" / "fair_comparison.yaml").read_text(encoding="utf-8"))
        fair["fixed_rules"]["datasets"] = ["Set5"]
        fair["pilot_determined"]["edge_top_q"] = 10
        fair["pilot_determined"]["representative_images"] = {"Set5": {"median": None, "worst": None}}
        fair["infer_benchmark"].update(lr_height=12, lr_width=16, sr_height=48, sr_width=64, warmup_runs=1, repeat_runs=1)

        hr_root = tmp_path / "hr_src"
        (hr_root / "Set5").mkdir(parents=True)
        flat = np.full((28, 28, 3), 128, dtype=np.uint8)
        Image.fromarray(flat).save(hr_root / "Set5" / "flat0.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["Set5"], "bicubic", "matlab_imresize_antialias")

        runs_dir = tmp_path / "runs"
        run_meta = {
            "run_id": "c0-bicubic-s0",
            "commit": "c0",
            "config_id": "bicubic",
            "seed": 0,
            "patch_pool_id": POOL,
            "switches": {"A": False, "B": False, "C": False, "Bp": False},
            "params": 0,
            "train_minutes": 0.0,
            "status": "complete",
        }
        evaluate_model(None, run_meta, data_dir, runs_dir / "c0-bicubic-s0", fair_comparison=fair, lpips_model=_StubLpips())

        summary, per_image = load_run_records(runs_dir)
        assert math.isinf(summary["psnr_y"].iloc[0]) and summary["psnr_y"].iloc[0] > 0
        assert math.isinf(per_image["edge_psnr"].iloc[0]) and per_image["edge_psnr"].iloc[0] > 0

        tables = compute_result_tables(summary, expected_seeds=SEEDS)
        bic = _result(tables, "bicubic", metric="psnr_y", dataset="Set5")
        assert bool(bic["is_complete"]) is True
        assert math.isinf(bic["mean"]) and bic["mean"] > 0
        assert math.isnan(bic["std"]) and math.isnan(bic["delta"])
        assert bic["delta_sign_consistent"] is pd.NA
        assert bic["delta_exceeds_seed_delta_std"] is pd.NA

    def test_reads_records_written_with_evaluator_record_fields(self, tmp_path: Path) -> None:
        """evaluator가 쓰는 필드(SummaryRecord·PerImageRecord)와 집계가 읽는 필드가 어긋나지 않아야 한다."""
        from dataclasses import asdict

        from sr_pipeline.evaluation.evaluator import PerImageRecord, SummaryRecord

        summary = SummaryRecord(
            run_id="c0-B-s0", commit="c0", config_id="B", seed=0, patch_pool_id=POOL, switches="B", dataset="Set5",
            psnr_y=30.0, ssim_y=0.9, psnr_rgb=28.0, edge_psnr=25.0, lpips=0.1, psnr_y_pre_rgb=30.1,
            params=24800, infer_ms_net=1.0, infer_ms_total=2.0, train_minutes=30.0, status="complete",
        )  # fmt: skip
        per_image = PerImageRecord(
            run_id="c0-B-s0", dataset="Set5", image_id="baby", psnr_y=30.0, ssim_y=0.9, psnr_rgb=28.0,
            edge_psnr=25.0, lpips=0.1,
        )  # fmt: skip
        run_dir = tmp_path / "c0-B-s0"
        run_dir.mkdir()
        pd.DataFrame([asdict(summary)]).to_csv(run_dir / "summary.csv", index=False)
        pd.DataFrame([asdict(per_image)]).to_csv(run_dir / "per_image.csv", index=False)

        loaded_summary, loaded_per_image = load_run_records(tmp_path)
        assert list(loaded_summary.columns) == list(asdict(summary))
        assert list(loaded_per_image.columns) == list(asdict(per_image))
        assert loaded_summary.psnr_y_pre_rgb.iloc[0] == pytest.approx(30.1)

    def test_numeric_looking_identifiers_stay_strings(self, tmp_path: Path) -> None:
        row = _row("baseline", 0, 30.0, commit="0123456", pool="1e5")
        self._write_run(tmp_path, row, [("0007", 30.0)])
        summary, per_image = load_run_records(tmp_path)
        assert summary.commit.iloc[0] == "0123456" and summary.patch_pool_id.iloc[0] == "1e5"
        assert per_image.image_id.iloc[0] == "0007"

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0, status="done"), [("im1", 30.0)])
        with pytest.raises(ValueError, match="status"):
            load_run_records(tmp_path)


def _per_image_rows(
    run_id: str,
    images: list[tuple[str, float]],
    *,
    dataset: str = "Set5",
    lpips_map: dict[str, float] | None = None,
) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "dataset": dataset,
            "image_id": image_id,
            "psnr_y": psnr_y,
            "ssim_y": 0.9,
            "psnr_rgb": 28.0,
            "edge_psnr": 25.0,
            "lpips": 0.1 if lpips_map is None else lpips_map[image_id],
        }
        for image_id, psnr_y in images
    ]


class TestPerImageDeltaAndCaseStudies:
    def test_per_image_delta_and_case_study_selection(self) -> None:
        """영상별 Δ 계산과 성공·실패 사례 영상 자동 선정이 규칙대로 동작해야 합니다."""
        summary = _frame(
            [_row("baseline", s, 30.0, dataset="BSD100") for s in SEEDS]
            + [_row("A", s, 30.5, dataset="BSD100") for s in SEEDS]
            + [_row("A", 0, 31.0, dataset="Urban100"), _row("baseline", 0, 30.0, dataset="Urban100")]
        )
        # BSD100 영상 3장 (시드 0, 1, 2):
        # img_best: A - baseline = [+1.0, +1.4, +1.2] -> 평균 +1.20
        # img_mid : A - baseline = [+0.2, +0.0, +0.4] -> 평균 +0.20
        # img_bad : A - baseline = [-0.5, -0.3, -0.4] -> 평균 -0.40
        per_image_list: list[dict] = []
        base_bsd = {0: [("img_best", 30.0), ("img_mid", 29.0), ("img_bad", 28.0)],
                    1: [("img_best", 30.2), ("img_mid", 29.5), ("img_bad", 28.1)],
                    2: [("img_best", 29.8), ("img_mid", 28.5), ("img_bad", 27.9)]}
        a_bsd = {0: [("img_best", 31.0), ("img_mid", 29.2), ("img_bad", 27.5)],
                 1: [("img_best", 31.6), ("img_mid", 29.5), ("img_bad", 27.8)],
                 2: [("img_best", 31.0), ("img_mid", 28.9), ("img_bad", 27.5)]}
        for s in SEEDS:
            per_image_list += _per_image_rows(f"c0-baseline-s{s}", base_bsd[s], dataset="BSD100")
            per_image_list += _per_image_rows(f"c0-A-s{s}", a_bsd[s], dataset="BSD100")
        per_image_list += _per_image_rows("c0-baseline-s0", [("u1", 25.0), ("u2", 26.0)], dataset="Urban100")
        per_image_list += _per_image_rows("c0-A-s0", [("u1", 25.5), ("u2", 24.5)], dataset="Urban100")
        per_image_df = pd.DataFrame(per_image_list, columns=PER_IMAGE_COLUMNS)

        deltas = compute_per_image_deltas(summary, per_image_df, expected_seeds=SEEDS)
        img_summary = deltas["per_image_summary"]
        bsd = img_summary[(img_summary.dataset == "BSD100") & (img_summary.config_id == "A") & (img_summary.metric == "psnr_y")]
        by_id = bsd.set_index("image_id")
        assert by_id.at["img_best", "mean_delta"] == pytest.approx(1.2)
        assert by_id.at["img_mid", "mean_delta"] == pytest.approx(0.2)
        assert by_id.at["img_bad", "mean_delta"] == pytest.approx(-0.4)
        assert by_id.at["img_best", "delta_sign"] == "positive"
        assert by_id.at["img_mid", "delta_sign"] == "mixed"
        assert by_id.at["img_bad", "delta_sign"] == "negative"
        assert by_id.at["img_best", "run_ids"] == "c0-A-s0,c0-A-s1,c0-A-s2"
        assert by_id.at["img_best", "baseline_run_ids"] == "c0-baseline-s0,c0-baseline-s1,c0-baseline-s2"

        cases = select_case_study_images(per_image_df, "A", summary_df=summary, expected_seeds=SEEDS)
        assert cases["BSD100"]["success"][0]["image_id"] == "img_best"
        assert cases["BSD100"]["success"][0]["mean_delta"] == pytest.approx(1.2)
        assert cases["BSD100"]["failure"][0]["image_id"] == "img_bad"
        assert cases["BSD100"]["failure"][0]["mean_delta"] == pytest.approx(-0.4)
        assert cases["Urban100"]["success"][0]["image_id"] == "u1"
        assert cases["Urban100"]["failure"][0]["image_id"] == "u2"

    def test_case_study_excludes_incomplete_runs_and_handles_lpips_direction_and_ties(self) -> None:
        summary = _frame(
            [_row("baseline", s, 30.0) for s in (0, 1)]
            + [_row("A", 0, 31.0), _row("A", 1, 50.0, status="incomplete")]
        )
        # 시드 1은 incomplete이므로 제외되고 시드 0만 비교된다.
        # LPIPS는 낮을수록 개선이므로 delta가 더 음수인 im_b(-0.08)가 성공, 양수인 im_a(+0.05)가 실패.
        # 동점 시 image_id 오름차순(im_a < im_c).
        per_image_df = pd.DataFrame(
            _per_image_rows("c0-baseline-s0", [("im_a", 30.0), ("im_b", 30.0), ("im_c", 30.0)],
                            lpips_map={"im_a": 0.20, "im_b": 0.20, "im_c": 0.20})
            + _per_image_rows("c0-A-s0", [("im_a", 31.0), ("im_b", 31.0), ("im_c", 31.0)],
                              lpips_map={"im_a": 0.25, "im_b": 0.12, "im_c": 0.25})
            + _per_image_rows("c0-baseline-s1", [("im_a", 30.0), ("im_b", 30.0), ("im_c", 30.0)],
                              lpips_map={"im_a": 0.20, "im_b": 0.20, "im_c": 0.20})
            + _per_image_rows("c0-A-s1", [("im_a", 20.0), ("im_b", 40.0), ("im_c", 20.0)],
                              lpips_map={"im_a": 0.01, "im_b": 0.99, "im_c": 0.01}),
            columns=PER_IMAGE_COLUMNS,
        )
        cases = select_case_study_images(per_image_df, "A", summary_df=summary, expected_seeds=SEEDS, metric="lpips")
        assert cases["Set5"]["success"][0]["image_id"] == "im_b"
        assert cases["Set5"]["success"][0]["mean_delta"] == pytest.approx(-0.08)
        assert cases["Set5"]["failure"][0]["image_id"] == "im_a"  # im_a(+0.05)와 im_c(+0.05) 동점 → image_id 오름차순
        assert cases["Set5"]["failure"][0]["run_ids"] == "c0-A-s0"

    def test_per_image_deltas_rejects_duplicate_image_rows(self) -> None:
        summary = _frame([_row("baseline", 0, 30.0), _row("A", 0, 31.0)])
        per_image_df = pd.DataFrame(
            _per_image_rows("c0-baseline-s0", [("im1", 30.0), ("im1", 30.0)])
            + _per_image_rows("c0-A-s0", [("im1", 31.0)]),
            columns=PER_IMAGE_COLUMNS,
        )
        with pytest.raises(ValueError, match="중복"):
            compute_per_image_deltas(summary, per_image_df, expected_seeds=SEEDS)

    def test_case_study_selection_keeps_rankable_infinite_deltas_and_handles_all_nan(self) -> None:
        summary = _frame([_row("baseline", 0, 30.0), _row("A", 0, 31.0)])
        per_image_df = pd.DataFrame(
            _per_image_rows(
                "c0-baseline-s0",
                [("im_perfect", 30.0), ("im_finite", 30.0), ("im_regressed", math.inf), ("im_both_inf", math.inf)],
            )
            + _per_image_rows(
                "c0-A-s0",
                [("im_perfect", math.inf), ("im_finite", 31.0), ("im_regressed", 29.0), ("im_both_inf", math.inf)],
            ),
            columns=PER_IMAGE_COLUMNS,
        )
        cases = select_case_study_images(per_image_df, "A", summary_df=summary, expected_seeds=SEEDS)
        assert cases["Set5"]["success"][0]["image_id"] == "im_perfect"
        assert math.isinf(cases["Set5"]["success"][0]["mean_delta"]) and cases["Set5"]["success"][0]["mean_delta"] > 0
        assert cases["Set5"]["failure"][0]["image_id"] == "im_regressed"
        assert math.isinf(cases["Set5"]["failure"][0]["mean_delta"]) and cases["Set5"]["failure"][0]["mean_delta"] < 0

        all_nan_df = pd.DataFrame(
            _per_image_rows("c0-baseline-s0", [("im_a", math.inf)]) + _per_image_rows("c0-A-s0", [("im_a", math.inf)]),
            columns=PER_IMAGE_COLUMNS,
        )
        assert select_case_study_images(all_nan_df, "A", summary_df=summary, expected_seeds=SEEDS) == {}


class TestReportTables:
    def test_formatted_result_table_masks_lpips_exceeds_and_flags_incomplete(self) -> None:
        rows = (
            [_row("baseline", s, 30.0) for s in SEEDS]
            + [_row("A", s, 30.4) for s in SEEDS]
            + [_row("B", 0, 30.2), _row("B", 1, 30.3, status="incomplete")]
        )
        for r in rows:
            if r["config_id"] == "A":
                r["lpips"] = 0.08  # baseline 0.1 -> delta -0.02 (음수가 개선)
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        formatted = format_result_report_table(tables["results"])

        a_lpips = formatted[(formatted.config_id == "A") & (formatted.metric == "lpips")].iloc[0]
        assert a_lpips["delta"] == pytest.approx(-0.02)
        assert a_lpips["delta_direction"] == "음수가 개선"
        assert a_lpips["delta_exceeds_seed_delta_std_display"] == "—"

        a_psnr = formatted[(formatted.config_id == "A") & (formatted.metric == "psnr_y")].iloc[0]
        assert a_psnr["completeness"] == "완료"
        assert a_psnr["delta_direction"] == "양수가 개선"

        b_psnr = formatted[(formatted.config_id == "B") & (formatted.metric == "psnr_y")].iloc[0]
        assert "미완료" in b_psnr["completeness"]
        assert "2" in b_psnr["completeness"]  # 누락 시드
        assert "1" in b_psnr["completeness"]  # incomplete 시드

        # 베이스라인이 미완료(시드 2 누락)일 때도 np.False_ is False 오류 없이 표시되어야 한다.
        inc_base_rows = [_row("baseline", 0, 30.0), _row("baseline", 1, 30.0)] + [_row("A", s, 31.0) for s in SEEDS]
        inc_formatted = format_result_report_table(compute_result_tables(_frame(inc_base_rows), expected_seeds=SEEDS)["results"])
        assert inc_formatted[inc_formatted.config_id == "A"].iloc[0]["completeness"] == "완료 (베이스라인 미완료)"

    def test_ablation_table_separates_a_contexts_and_b_decomposition(self) -> None:
        # 손계산용 Set5 psnr_y:
        # baseline: [30.0, 30.0, 30.0] -> mean 30.0
        # A       : [30.4, 30.5, 30.3] -> mean 30.4 -> A-baseline = +0.4 (seed deltas [0.4, 0.5, 0.3], std 0.1)
        # Bp      : [30.1, 30.1, 30.1] -> mean 30.1 -> Bp-baseline = +0.1 (정렬 효과)
        # B       : [30.6, 30.7, 30.5] -> mean 30.6 -> B-Bp = +0.5 (입력·용량 효과), B-baseline = +0.6
        # AB      : [30.7, 30.8, 30.6] -> mean 30.7 -> AB-B = +0.1 (Y 맥락 A 효과)
        values = {
            "baseline": (30.0, 30.0, 30.0),
            "A": (30.4, 30.5, 30.3),
            "Bp": (30.1, 30.1, 30.1),
            "B": (30.6, 30.7, 30.5),
            "AB": (30.7, 30.8, 30.6),
        }
        summary = _frame([_row(cfg, s, v) for cfg, vs in values.items() for s, v in zip(SEEDS, vs)])
        ablation = build_ablation_table(summary, expected_seeds=SEEDS)

        sub = ablation[(ablation.dataset == "Set5") & (ablation.metric == "psnr_y")].set_index("contrast")
        assert sub.at["A_rgb_context", "delta"] == pytest.approx(0.4)
        assert sub.at["A_rgb_context", "delta_sign"] == "positive"
        assert bool(sub.at["A_rgb_context", "delta_exceeds_seed_delta_std"]) is True
        assert sub.at["A_y_context", "delta"] == pytest.approx(0.1)
        assert sub.at["B_alignment", "delta"] == pytest.approx(0.1)
        assert sub.at["B_input_capacity", "delta"] == pytest.approx(0.5)
        assert sub.at["B_total", "delta"] == pytest.approx(0.6)
        assert sub.at["A_y_context", "target_run_ids"] == "c0-AB-s0,c0-AB-s1,c0-AB-s2"
        assert sub.at["A_y_context", "reference_run_ids"] == "c0-B-s0,c0-B-s1,c0-B-s2"

    def test_progress_and_training_time_and_b_pre_rgb_tables(self) -> None:
        rows = [
            _row("bicubic", 0, 28.0),
            _row("baseline", 0, 30.0),
            _row("baseline", 1, 30.1),
            _row("baseline", 2, 29.9),
            _row("B", 0, 30.5),
            _row("B", 1, 30.6, status="incomplete"),
        ]
        for r in rows:
            if r["config_id"] == "bicubic":
                r["train_minutes"] = 0.0
                r["params"] = 0
            elif r["config_id"] == "B":
                r["psnr_y_pre_rgb"] = r["psnr_y"] + 0.05
                r["params"] = 24752
                r["train_minutes"] = 25.0 if r["seed"] == 0 else 185.0
            else:
                r["params"] = 37200
                r["train_minutes"] = 30.0

        summary = _frame(rows)
        progress = build_progress_table(
            summary,
            expected_configs=["bicubic", "baseline", "A", "B", "AC"],
            expected_seeds=SEEDS,
            promoted_configs=["AC"],
        ).set_index("config_id")
        assert progress.at["bicubic", "status"] == "완료"
        assert progress.at["baseline", "status"] == "완료"
        assert progress.at["A", "status"] == "미실행"
        assert progress.at["B", "status"] == "미완료"
        assert "incomplete" in progress.at["B", "seed_1"]
        assert "missing" in progress.at["B", "seed_2"]
        assert progress.at["AC", "status"] == "보류(미승격)"

        train_logs = {
            "c0-baseline-s0": {"epochs_done": 10, "epochs_total": 10, "epoch_losses": [0.2, 0.1]},
            "c0-B-s1": {"epochs_done": 6, "epochs_total": 10, "epoch_losses": [0.3, 0.2]},
        }
        times = build_training_time_table(summary, train_logs=train_logs, max_hours=3.0)
        b1 = times[times["run_id"] == "c0-B-s1"].iloc[0]
        assert bool(b1["within_max_hours"]) is False
        assert b1["epochs_done"] == 6 and b1["epochs_total"] == 10
        assert b1["final_epoch_loss"] == pytest.approx(0.2)

        pre_rgb = build_b_pre_rgb_diagnostics_table(summary)
        assert len(pre_rgb) == 1
        assert pre_rgb.iloc[0]["config_id"] == "B"
        assert pre_rgb.iloc[0]["diff_pre_minus_post"] == pytest.approx(0.05)


class TestExportReportAssets:
    def _write_full_run(
        self,
        runs_dir: Path,
        config_id: str,
        seed: int,
        *,
        psnr_shift: float = 0.0,
        status: str = "complete",
        with_images: bool = True,
        with_train_log: bool = True,
        with_weights: bool = False,
        extra_params: int = 37200,
    ) -> str:
        import json
        import numpy as np
        import torch
        from PIL import Image
        from sr_pipeline.models.espcn import ESPCN

        run_id = f"c0-{config_id}-s{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        datasets = {
            "Set5": [("img0", 30.0 + psnr_shift), ("img1", 28.0 + psnr_shift)],
            "BSD100": [("b0", 29.0 + psnr_shift), ("b1", 29.5 + psnr_shift * 0.5), ("b2", 28.5 - 0.1)],
            "Urban100": [("u0", 27.0 + psnr_shift), ("u1", 26.5 + psnr_shift * 2.0)],
        }
        summary_rows = []
        per_image_rows = []
        for ds, imgs in datasets.items():
            mean_psnr = float(np.mean([v for _, v in imgs]))
            r = _row(config_id, seed, mean_psnr, dataset=ds, status=status)
            r["params"] = 0 if config_id == "bicubic" else extra_params
            r["train_minutes"] = 0.0 if config_id == "bicubic" else 12.0 + seed
            if config_id == "B":
                r["psnr_y_pre_rgb"] = mean_psnr + 0.03
            summary_rows.append(r)
            for img_id, val in imgs:
                per_image_rows.append(
                    {"run_id": run_id, "dataset": ds, "image_id": img_id, "psnr_y": val,
                     "ssim_y": 0.9, "psnr_rgb": val - 1.0, "edge_psnr": val - 2.0, "lpips": 0.1 - 0.01 * psnr_shift}
                )
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(run_dir / "summary.csv", index=False)
        pd.DataFrame(per_image_rows, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
        if with_train_log and config_id != "bicubic":
            (run_dir / "train_log.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config_id": config_id,
                        "seed": seed,
                        "status": status,
                        "epochs_done": 3 if status == "complete" else 2,
                        "epochs_total": 3,
                        "train_minutes": 12.0 + seed,
                        "epoch_losses": [0.25 - 0.02 * psnr_shift, 0.18 - 0.02 * psnr_shift, 0.14 - 0.02 * psnr_shift]
                        if status == "complete"
                        else [0.25, 0.20],
                    }
                ),
                encoding="utf-8",
            )
        if with_images and seed == 0:
            img_dir = run_dir / "images" / "Set5"
            img_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(10 + hash(config_id) % 100)
            before = np.full((32, 32, 3), 100, dtype=np.uint8)
            hr = np.full((32, 32, 3), 130, dtype=np.uint8)
            after = np.clip(
                before.astype(np.int16) + int(10 * psnr_shift) + rng.integers(0, 5, (32, 32, 3)), 0, 255
            ).astype(np.uint8)
            Image.fromarray(np.concatenate([before, after, hr], axis=1)).save(img_dir / "img0_median.png")
            roi = np.repeat(np.repeat(np.concatenate([before[:8, :8], after[:8, :8], hr[:8, :8]], axis=1), 4, 0), 4, 1)
            Image.fromarray(roi).save(img_dir / "img0_median_roi.png")
        if with_weights and config_id != "bicubic":
            ckpt_dir = runs_dir.parent / "ckpts" / run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model = ESPCN(use_y_channel=(config_id in ("B", "AB")), use_global_residual=(config_id == "C"))
            torch.save(model.state_dict(), ckpt_dir / "weights.pt")
        return run_id

    def test_export_report_assets_writes_all_tables_and_figures(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        self._write_full_run(runs_dir, "bicubic", 0, psnr_shift=-1.0)
        for seed in SEEDS:
            self._write_full_run(runs_dir, "baseline", seed, psnr_shift=0.0)
            self._write_full_run(runs_dir, "A", seed, psnr_shift=0.5)
            self._write_full_run(runs_dir, "Bp", seed, psnr_shift=0.2)
            self._write_full_run(runs_dir, "B", seed, psnr_shift=0.7, extra_params=24752)
            self._write_full_run(runs_dir, "AB", seed, psnr_shift=0.9, extra_params=24752)

        manifest = export_report_assets(runs_dir, out_dir)

        tables_dir = out_dir / "tables"
        figures_dir = out_dir / "figures"
        for stem in ("results", "ablation", "case_studies", "training_times", "progress", "b_pre_rgb_diagnostics"):
            assert (tables_dir / f"{stem}.csv").is_file(), f"missing {stem}.csv"
            assert (tables_dir / f"{stem}.md").is_file(), f"missing {stem}.md"
        assert (tables_dir / "seed_deltas.csv").is_file()
        assert (tables_dir / "per_image_summary.csv").is_file()
        results_md = (tables_dir / "results.md").read_text(encoding="utf-8")
        assert "음수가 개선" in results_md
        lpips_lines = [line for line in results_md.splitlines() if "| lpips |" in line]
        assert lpips_lines
        for line in lpips_lines:
            # LPIPS 행의 delta_exceeds_seed_delta_std 칸은 — 여야 하고, 예/아니오가 노출되면 안 된다.
            cells = [c.strip() for c in line.strip("|").split("|")]
            header = [c.strip() for c in results_md.splitlines()[2].strip("|").split("|")]
            row_map = dict(zip(header, cells))
            assert row_map["delta_exceeds_seed_delta_std"] == "—"
            assert "delta_exceeds_seed_delta_std_display" not in row_map

        assert (figures_dir / "per_image_delta_distribution_BSD100.png").is_file()
        assert (figures_dir / "per_image_delta_distribution_Urban100.png").is_file()
        assert (figures_dir / "learning_curves.png").is_file()
        assert (figures_dir / "comparison_Set5_img0_median.png").is_file()
        assert (figures_dir / "error_maps_Set5_img0_median.png").is_file()
        assert (out_dir / "manifest.json").is_file()
        assert len(manifest["run_ids"]) == 16

    def test_export_report_assets_renders_case_study_figures_when_data_and_checkpoints_provided(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image
        from sr_pipeline.data.prepare import prepare_testsets

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        self._write_full_run(runs_dir, "bicubic", 0, psnr_shift=-1.0, with_images=False)
        for seed in SEEDS:
            self._write_full_run(runs_dir, "baseline", seed, psnr_shift=0.0, with_images=False, with_weights=(seed == 0))
            self._write_full_run(runs_dir, "A", seed, psnr_shift=0.5, with_images=False, with_weights=(seed == 0))

        hr_root = tmp_path / "hr_src"
        for ds, ids in (("Set5", ("img0", "img1")), ("BSD100", ("b0", "b1", "b2")), ("Urban100", ("u0", "u1"))):
            (hr_root / ds).mkdir(parents=True)
            for idx, img_id in enumerate(ids):
                arr = np.full((32, 32, 3), 100 + 20 * idx, dtype=np.uint8)
                arr[8:24, 8:24] = 200
                Image.fromarray(arr).save(hr_root / ds / f"{img_id}.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["Set5", "BSD100", "Urban100"], "bicubic", "matlab_imresize_antialias")

        export_report_assets(runs_dir, out_dir, data_dir=data_dir, checkpoint_dir=tmp_path / "ckpts")

        figures_dir = out_dir / "figures"
        # BSD100에서 b0가 성공(+0.5), b2가 실패(-0.1 - shift 기준)로 자동 선정되어 비교·오차 지도가 생성된다.
        assert (figures_dir / "case_study_comparison_A_BSD100_b0_success.png").is_file()
        assert (figures_dir / "case_study_error_maps_A_BSD100_b0_success.png").is_file()
        assert (figures_dir / "case_study_comparison_A_BSD100_b2_failure.png").is_file()
        assert (figures_dir / "case_study_error_maps_A_BSD100_b2_failure.png").is_file()

    def test_export_report_assets_skips_case_study_figure_when_target_weights_missing(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image
        from sr_pipeline.data.prepare import prepare_testsets

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        self._write_full_run(runs_dir, "bicubic", 0, psnr_shift=-1.0, with_images=False)
        for seed in SEEDS:
            self._write_full_run(runs_dir, "baseline", seed, psnr_shift=0.0, with_images=False, with_weights=(seed == 0))
            self._write_full_run(runs_dir, "A", seed, psnr_shift=0.5, with_images=False, with_weights=False)

        hr_root = tmp_path / "hr_src"
        for ds, ids in (("Set5", ("img0", "img1")), ("BSD100", ("b0", "b1", "b2")), ("Urban100", ("u0", "u1"))):
            (hr_root / ds).mkdir(parents=True)
            for idx, img_id in enumerate(ids):
                arr = np.full((32, 32, 3), 100 + 20 * idx, dtype=np.uint8)
                arr[8:24, 8:24] = 200
                Image.fromarray(arr).save(hr_root / ds / f"{img_id}.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["Set5", "BSD100", "Urban100"], "bicubic", "matlab_imresize_antialias")

        manifest = export_report_assets(runs_dir, out_dir, data_dir=data_dir, checkpoint_dir=tmp_path / "ckpts")
        assert not any(name.startswith("case_study_") for name in manifest["figures"])
        assert "case_study_figures_skipped_reason" in manifest
        assert manifest["skipped_case_study_figures"]

    def test_export_report_assets_separates_plots_by_patch_pool(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        for pool, offset in (("pool_1", 0.0), ("pool_2", 1.0)):
            for cfg, shift in (("baseline", 0.0), ("A", 0.5)):
                for seed in SEEDS:
                    run_id = f"{pool}-{cfg}-s{seed}"
                    run_dir = runs_dir / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    rows = []
                    per_img = []
                    for ds in ("BSD100", "Urban100"):
                        r = _row(cfg, seed, 30.0 + offset + shift, dataset=ds, pool=pool, commit=pool)
                        r["run_id"] = run_id
                        rows.append(r)
                        for idx, img_id in enumerate(("im0", "im1")):
                            per_img.append(
                                {"run_id": run_id, "dataset": ds, "image_id": img_id,
                                 "psnr_y": 30.0 + offset + shift + 0.1 * idx, "ssim_y": 0.9,
                                 "psnr_rgb": 28.0, "edge_psnr": 25.0, "lpips": 0.1}
                            )
                    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(run_dir / "summary.csv", index=False)
                    pd.DataFrame(per_img, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
                    import json
                    (run_dir / "train_log.json").write_text(
                        json.dumps({"run_id": run_id, "config_id": cfg, "seed": seed, "patch_pool_id": pool,
                                    "epochs_done": 2, "epochs_total": 2, "epoch_losses": [0.2 + offset, 0.1 + offset]}),
                        encoding="utf-8",
                    )

        manifest = export_report_assets(runs_dir, out_dir)
        figures_dir = out_dir / "figures"
        for pool in ("pool_1", "pool_2"):
            assert (figures_dir / f"per_image_delta_distribution_{pool}_BSD100.png").is_file()
            assert (figures_dir / f"per_image_delta_distribution_{pool}_Urban100.png").is_file()
            assert (figures_dir / f"learning_curves_{pool}.png").is_file()
        assert "learning_curves.png" not in manifest["figures"]
        assert "per_image_delta_distribution_BSD100.png" not in manifest["figures"]

    def test_export_report_assets_uses_paired_seed_checkpoints_and_records_skipped(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image
        from sr_pipeline.data.prepare import prepare_testsets

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        # baseline은 시드 0, 1 모두 체크포인트가 있고, A는 시드 1만 complete+체크포인트(시드 0은 incomplete).
        # B는 시드 1만 complete+체크포인트이지만 baseline 시드 1 체크포인트가 없는 상황 대신:
        # C는 시드 2만 complete(체크포인트 없음) -> 건너뜀 목록에 기록, A는 반드시 시드 1 쌍(c0-A-s1 vs c0-baseline-s1)을 써야 함.
        self._write_full_run(runs_dir, "baseline", 0, psnr_shift=0.0, with_images=False, with_weights=True)
        self._write_full_run(runs_dir, "baseline", 1, psnr_shift=0.0, with_images=False, with_weights=True)
        self._write_full_run(runs_dir, "A", 0, psnr_shift=0.5, status="incomplete", with_images=False, with_weights=True)
        self._write_full_run(runs_dir, "A", 1, psnr_shift=0.5, status="complete", with_images=False, with_weights=True)
        self._write_full_run(runs_dir, "C", 1, psnr_shift=0.3, status="complete", with_images=False, with_weights=False)

        hr_root = tmp_path / "hr_src"
        for ds, ids in (("Set5", ("img0", "img1")), ("BSD100", ("b0", "b1", "b2")), ("Urban100", ("u0", "u1"))):
            (hr_root / ds).mkdir(parents=True)
            for idx, img_id in enumerate(ids):
                arr = np.full((32, 32, 3), 100 + 20 * idx, dtype=np.uint8)
                arr[8:24, 8:24] = 200
                Image.fromarray(arr).save(hr_root / ds / f"{img_id}.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["Set5", "BSD100", "Urban100"], "bicubic", "matlab_imresize_antialias")

        manifest = export_report_assets(runs_dir, out_dir, data_dir=data_dir, checkpoint_dir=tmp_path / "ckpts")
        rendered_a = [item for item in manifest["case_study_figures"] if item["config_id"] == "A"]
        assert rendered_a
        assert all(item["seed"] == 1 for item in rendered_a)
        assert all(item["run_id"] == "c0-A-s1" for item in rendered_a)
        assert all(item["baseline_run_id"] == "c0-baseline-s1" for item in rendered_a)
        skipped_c = [item for item in manifest["skipped_case_study_figures"] if item["config_id"] == "C"]
        assert skipped_c
        assert "c0-C-s1" in skipped_c[0]["reason"]

    def test_export_report_assets_handles_all_nan_case_study_deltas(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        for cfg in ("baseline", "A"):
            run_id = f"c0-{cfg}-s0"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            row = _row(cfg, 0, math.inf, dataset="BSD100")
            pd.DataFrame([row], columns=SUMMARY_COLUMNS).to_csv(run_dir / "summary.csv", index=False)
            per_img = [{"run_id": run_id, "dataset": "BSD100", "image_id": "flat0", "psnr_y": math.inf,
                        "ssim_y": 1.0, "psnr_rgb": math.inf, "edge_psnr": math.inf, "lpips": 0.0}]
            pd.DataFrame(per_img, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
            import json
            (run_dir / "train_log.json").write_text(
                json.dumps({"run_id": run_id, "config_id": cfg, "seed": 0, "patch_pool_id": POOL,
                            "epochs_done": 1, "epochs_total": 1, "epoch_losses": [0.0]}),
                encoding="utf-8",
            )

        manifest = export_report_assets(runs_dir, out_dir)
        assert (out_dir / "tables" / "results.csv").is_file()
        assert (out_dir / "tables" / "case_studies.csv").is_file()
        assert (out_dir / "figures" / "learning_curves.png").is_file()
        assert manifest["run_ids"] == ["c0-A-s0", "c0-baseline-s0"]

    def test_case_study_figure_title_distinguishes_seed_delta_from_mean_delta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np
        from PIL import Image
        import sr_pipeline.reporting.aggregate as agg_mod
        from sr_pipeline.data.prepare import prepare_testsets

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        # 시드 0 Δ = +1.0 dB, 시드 1 Δ = +3.0 dB -> 평균 Δ = +2.0 dB (시드 0 그림에는 둘을 구분 표기해야 함).
        for seed, delta in ((0, 1.0), (1, 3.0)):
            for cfg, val in (("baseline", 30.0), ("A", 30.0 + delta)):
                run_id = f"c0-{cfg}-s{seed}"
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([_row(cfg, seed, val, dataset="BSD100")], columns=SUMMARY_COLUMNS).to_csv(
                    run_dir / "summary.csv", index=False
                )
                per_img = [
                    {"run_id": run_id, "dataset": "BSD100", "image_id": "b0", "psnr_y": val,
                     "ssim_y": 0.9, "psnr_rgb": 28.0, "edge_psnr": 25.0, "lpips": 0.1},
                    {"run_id": run_id, "dataset": "BSD100", "image_id": "b1", "psnr_y": 30.0,
                     "ssim_y": 0.9, "psnr_rgb": 28.0, "edge_psnr": 25.0, "lpips": 0.1},
                ]
                pd.DataFrame(per_img, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
                ckpt_dir = tmp_path / "ckpts" / run_id
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                from sr_pipeline.models.espcn import ESPCN
                import torch
                torch.save(ESPCN().state_dict(), ckpt_dir / "weights.pt")

        hr_root = tmp_path / "hr_src"
        (hr_root / "BSD100").mkdir(parents=True)
        for img_id in ("b0", "b1"):
            Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(hr_root / "BSD100" / f"{img_id}.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["BSD100"], "bicubic", "matlab_imresize_antialias")

        captured_titles: list[str] = []
        orig_render = agg_mod._render_multi_config_panels

        def _spy_render(*args, **kwargs):
            captured_titles.append(kwargs["title"])
            return orig_render(*args, **kwargs)

        monkeypatch.setattr(agg_mod, "_render_multi_config_panels", _spy_render)
        manifest = export_report_assets(runs_dir, out_dir, data_dir=data_dir, checkpoint_dir=tmp_path / "ckpts")

        b0_rec = [r for r in manifest["case_study_figures"] if r["image_id"] == "b0"][0]
        assert b0_rec["seed"] == 0
        assert b0_rec["seed_delta"] == pytest.approx(1.0)
        assert b0_rec["mean_delta"] == pytest.approx(2.0)
        b0_title = [t for t in captured_titles if "b0" in t][0]
        assert "seed=0 delta=+1.000 dB" in b0_title
        assert "mean delta=+2.000 dB" in b0_title

    def test_distribution_win_fraction_includes_infinite_deltas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from sr_pipeline.reporting.aggregate import _plot_per_image_delta_distributions

        df = pd.DataFrame(
            [
                {"patch_pool_id": POOL, "dataset": "BSD100", "config_id": "A", "image_id": "im0", "metric": "psnr_y", "mean_delta": 1.0},
                {"patch_pool_id": POOL, "dataset": "BSD100", "config_id": "A", "image_id": "im1", "metric": "psnr_y", "mean_delta": -math.inf},
            ]
        )
        texts_seen: list[str] = []
        import matplotlib.axes

        orig_text = matplotlib.axes.Axes.text

        def _spy_text(self, x, y, s, *args, **kwargs):
            texts_seen.append(str(s))
            return orig_text(self, x, y, s, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)
        files = _plot_per_image_delta_distributions(df, tmp_path, config_order=["A"])
        assert files == ["per_image_delta_distribution_BSD100.png"]
        assert texts_seen == ["50%"]

    def test_rejects_mismatched_patch_pool_id_between_train_log_and_summary(self, tmp_path: Path) -> None:
        import json

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        self._write_full_run(runs_dir, "baseline", 0, psnr_shift=0.0, with_images=False)
        self._write_full_run(runs_dir, "A", 0, psnr_shift=0.5, with_images=False)
        bad_log_path = runs_dir / "c0-A-s0" / "train_log.json"
        bad_log = json.loads(bad_log_path.read_text(encoding="utf-8"))
        bad_log["patch_pool_id"] = "pool_other"
        bad_log_path.write_text(json.dumps(bad_log), encoding="utf-8")

        with pytest.raises(ValueError, match="patch_pool_id"):
            export_report_assets(runs_dir, out_dir)

    def test_manifest_json_serializes_infinite_deltas_as_strict_json_strings(self, tmp_path: Path) -> None:
        import json
        import numpy as np
        import torch
        from PIL import Image
        from sr_pipeline.data.prepare import prepare_testsets
        from sr_pipeline.models.espcn import ESPCN

        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "aggregated"
        for cfg, img_psnr in (("baseline", 30.0), ("A", math.inf)):
            run_id = f"c0-{cfg}-s0"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([_row(cfg, 0, img_psnr, dataset="BSD100")], columns=SUMMARY_COLUMNS).to_csv(
                run_dir / "summary.csv", index=False
            )
            per_img = [
                {"run_id": run_id, "dataset": "BSD100", "image_id": "b0", "psnr_y": img_psnr,
                 "ssim_y": 1.0 if math.isinf(img_psnr) else 0.9, "psnr_rgb": img_psnr, "edge_psnr": img_psnr, "lpips": 0.0},
                {"run_id": run_id, "dataset": "BSD100", "image_id": "b1", "psnr_y": 30.0,
                 "ssim_y": 0.9, "psnr_rgb": 28.0, "edge_psnr": 25.0, "lpips": 0.1},
            ]
            pd.DataFrame(per_img, columns=PER_IMAGE_COLUMNS).to_csv(run_dir / "per_image.csv", index=False)
            ckpt_dir = tmp_path / "ckpts" / run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ESPCN().state_dict(), ckpt_dir / "weights.pt")

        hr_root = tmp_path / "hr_src"
        (hr_root / "BSD100").mkdir(parents=True)
        for img_id in ("b0", "b1"):
            Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(hr_root / "BSD100" / f"{img_id}.png")
        data_dir = tmp_path / "data"
        prepare_testsets(hr_root, data_dir, 4, ["BSD100"], "bicubic", "matlab_imresize_antialias")

        manifest = export_report_assets(runs_dir, out_dir, data_dir=data_dir, checkpoint_dir=tmp_path / "ckpts")

        def _reject_nonstandard(token: str) -> None:
            raise ValueError(f"Non-standard JSON constant: {token}")

        parsed = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"), parse_constant=_reject_nonstandard)
        b0_rec = [r for r in parsed["case_study_figures"] if r["image_id"] == "b0"][0]
        assert b0_rec["seed_delta"] == "+inf"
        assert b0_rec["mean_delta"] == "+inf"
        b1_rec = [r for r in parsed["case_study_figures"] if r["image_id"] == "b1"][0]
        assert isinstance(b1_rec["seed_delta"], float) and b1_rec["seed_delta"] == pytest.approx(0.0)
        assert manifest == parsed
