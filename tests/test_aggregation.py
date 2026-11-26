"""자동 집계 보조 경계 테스트 (SPEC.md Testing Decisions §3).

입력은 손으로 계산한 값을 넣은 고정된 가짜 실행 기록이다. 기대값은 코드가 아니라 아래 주석의 손계산에서 온다.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from sr_pipeline.reporting.aggregate import (
    compute_result_tables,
    load_run_records,
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

    def test_all_negative_seed_deltas_are_a_consistent_direction(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS] + [_row("A", 0, 29.5), _row("A", 1, 29.8), _row("A", 2, 29.9)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        # 시드별 Δ = [-0.5, -0.2, -0.1] -> 평균 -0.2667, 표준편차 sqrt((0.0544+0.0044+0.0278)/2)=0.2082
        assert a["delta"] == pytest.approx(-0.8 / 3)
        assert a["delta_sign"] == "negative"
        assert bool(a["delta_sign_consistent"]) is True
        assert bool(a["delta_exceeds_seed_delta_std"]) is True

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

    def test_incomplete_status_run_is_marked_incomplete(self) -> None:
        rows = [_row("baseline", s, 30.0) for s in SEEDS]
        rows += [_row("A", 0, 31.0), _row("A", 1, 31.0, status="incomplete"), _row("A", 2, 31.0)]
        tables = compute_result_tables(_frame(rows), expected_seeds=SEEDS)
        a = _result(tables, "A")
        assert bool(a["is_complete"]) is False
        assert a["missing_seeds"] == ""
        assert a["incomplete_seeds"] == "1"

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

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_compute_rejects_non_finite_metric(self, value: float) -> None:
        frame = _fake_records()
        frame.loc[0, "lpips"] = value
        with pytest.raises(ValueError, match="lpips"):
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
        [("psnr_y", "NaN"), ("lpips", "nan"), ("ssim_y", "inf"), ("params", "abc"), ("psnr_y_pre_rgb", "inf")],
    )
    def test_rejects_non_finite_or_non_numeric_summary_value(self, tmp_path: Path, column: str, value: str) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "summary.csv"
        pd.read_csv(path).assign(**{column: value}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            load_run_records(tmp_path)

    def test_rejects_non_finite_per_image_value(self, tmp_path: Path) -> None:
        self._write_run(tmp_path, _row("baseline", 0, 30.0), [("im1", 30.0)])
        path = tmp_path / "c0-baseline-s0" / "per_image.csv"
        pd.read_csv(path).assign(edge_psnr="inf").to_csv(path, index=False)
        with pytest.raises(ValueError, match="edge_psnr"):
            load_run_records(tmp_path)

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


@pytest.mark.skip(reason="T6.3: 영상별 Δ·사례 영상 선정은 집계 산출물 Task에서 구현")
def test_per_image_delta_and_case_study_selection() -> None:
    """영상별 Δ 계산과 성공·실패 사례 영상 자동 선정이 규칙대로 동작해야 합니다."""
