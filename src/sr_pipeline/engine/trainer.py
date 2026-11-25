"""구성+시드 단일 실행 파이프라인 (SPEC.md / ADR-0001 / ADR-0005).

실행 1회가 결과 수치의 최소 출처 단위입니다:
- run_id = <commit>-<config_id>-s<seed>
- 커밋되지 않은 변경 감지(경고 또는 거부)
- 에폭 기반 학습 (3시간 상한 초과 시 incomplete 기록)
- 중간/마지막 에폭 체크포인트 저장 및 동일 run_id 재개 지원
- 마지막 에폭(평가 체크포인트)으로 통합 평가 실행 및 실행 기록 저장
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sr_pipeline.common.seed import set_model_seed
from sr_pipeline.config.schema import ResolvedRunConfig, Switches
from sr_pipeline.data.dataset import PatchPoolDataset
from sr_pipeline.data.sampler import SeededEpochPatchSampler, check_augmentation
from sr_pipeline.diagnosis.provenance import resolve_code_commit
from sr_pipeline.evaluation.benchmark import count_parameters
from sr_pipeline.evaluation.evaluator import evaluate_model
from sr_pipeline.losses.sr_loss import SRLoss
from sr_pipeline.models.espcn import ESPCN, SCALE

STATE_FILENAME = "state.pt"
WEIGHTS_FILENAME = "weights.pt"
TRAIN_LOG_FILENAME = "train_log.json"
SUPPORTED_OPTIMIZERS = ("adam",)
SUPPORTED_SCHEDULERS = ("none",)
COMPLETE = "complete"
INCOMPLETE = "incomplete"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_run_id(commit: str, config_id: str, seed: int) -> str:
    """SPEC.md 형식의 실행 ID(`<commit>-<config_id>-s<seed>`)를 생성합니다."""
    return f"{commit}-{config_id}-s{seed}"


def check_git_clean(repo_root: Path, allow_dirty: bool = False) -> str:
    """`repo_root` 저장소의 HEAD 커밋을 반환합니다.

    `src/`·`scripts/`·`configs/`에 커밋되지 않은 변경이 있으면 `RuntimeError`로 거부한다.
    `allow_dirty=True`면 거부하는 대신 `"<커밋>-dirty"`를 반환해, 실행 ID만 보고도 깨끗하지 않은 실행임을 알 수 있다.
    파일럿 진단과 같은 규칙(`diagnosis.provenance`)을 쓴다.
    """
    commit = resolve_code_commit(repo_root, allow_dirty=allow_dirty)
    if commit.endswith("-dirty"):
        warnings.warn(
            "코드·설정에 커밋되지 않은 변경이 있는 채로 실행합니다. 이 실행의 커밋은 실제로 돈 코드와 일치하지 않습니다.",
            stacklevel=2,
        )
    return commit


@dataclass(frozen=True)
class TrainResult:
    """`train_run`의 결과. 마지막으로 끝낸 에폭의 모델과 기록 값을 담는다(경로는 담지 않는다)."""

    run_id: str
    status: str  # COMPLETE | INCOMPLETE
    epochs_done: int
    epochs_total: int
    epoch_losses: list[float]
    train_minutes: float
    params: int
    model: ESPCN


def build_model_and_loss(switches: Switches, fair_comparison: dict[str, Any]) -> tuple[ESPCN, SRLoss]:
    """개선 스위치에서 모델과 손실을 만듭니다. 모든 스위치가 꺼진 구성이 곧 베이스라인입니다.

    A → `SRLoss(use_edge_loss=True, edge_loss_lambda=λ)`(λ는 `pilot_determined.edge_loss_lambda`),
    B → `ESPCN(use_y_channel=True)` + `SRLoss(use_y_loss=True)`, C → `ESPCN(use_global_residual=True)`,
    Bp(B′) → `SRLoss(use_y_loss=True)`. B′는 RGB 입출력에 손실만 Y에서 계산하는 통제 구성이라 B와 함께 켤 수 없다.
    """
    if switches.B and switches.Bp:
        raise ValueError("스위치 B와 Bp(B′)는 함께 켤 수 없습니다: B′는 B의 입력·용량 변화를 뺀 통제 구성입니다.")
    model = ESPCN(use_y_channel=switches.B, use_global_residual=switches.C)
    loss_fn = SRLoss(
        use_edge_loss=switches.A,
        use_y_loss=switches.B or switches.Bp,
        edge_loss_lambda=fair_comparison["pilot_determined"]["edge_loss_lambda"] if switches.A else 0.0,
    )
    return model, loss_fn


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_positive_int(value: Any, name: str) -> None:
    if not _is_int(value) or value < 1:
        raise ValueError(f"{name}은(는) 1 이상의 정수여야 합니다: {value!r}")


def _check_choice(value: Any, name: str, supported: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in supported:
        raise ValueError(f"{name}은(는) {supported} 중 하나여야 합니다: {value!r}")


def _check_training_settings(pilot: dict[str, Any]) -> None:
    for name in ("patch_pool_size", "patches_per_epoch", "epochs", "batch_size"):
        _check_positive_int(pilot[name], name)
    learning_rate = pilot["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError(f"learning_rate는 숫자여야 합니다: {learning_rate!r}")
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError(f"learning_rate는 0보다 큰 유한한 수여야 합니다: {learning_rate!r}")
    _check_choice(pilot["optimizer"], "optimizer", SUPPORTED_OPTIMIZERS)
    _check_choice(pilot["scheduler"], "scheduler", SUPPORTED_SCHEDULERS)
    check_augmentation(pilot["augmentation"])


def _settings_fingerprint(fair_comparison: dict[str, Any]) -> str:
    """재개할 때 학습 조건이 그대로인지 확인하는 해시. 공정 비교 조건 전체를 정렬된 JSON으로 만들어 쓴다."""
    canonical = json.dumps(fair_comparison, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) uint8을 [0, 1] float32 (B, 3, H, W)로 바꾼다."""
    return torch.from_numpy(batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)


def _atomic_save(obj: Any, path: Path) -> None:
    """다 쓴 뒤 이름을 바꿔, 저장 중 실패해도 반쯤 쓴 파일이 그 이름으로 남지 않게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        torch.save(obj, partial)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _write_train_log(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def train_run(
    run_config: ResolvedRunConfig,
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    commit: str,
    resume: bool = False,
    device: str = "cpu",
    clock: Callable[[], float] = time.perf_counter,
) -> TrainResult:
    """확정된 실행 조건으로 에폭 기반 학습을 하고 체크포인트와 학습 기록을 저장합니다.

    체크포인트는 `<checkpoint_dir>/<run_id>/`(`state.pt`, `weights.pt`), 학습 기록은
    `<output_dir>/<run_id>/train_log.json`에 쓴다.
    """
    if not run_config.trainable:
        raise ValueError(f"구성 {run_config.config_id}는 학습하지 않는 구성이라 학습할 수 없습니다.")
    fair = run_config.fair_comparison
    pilot = fair["pilot_determined"]
    _check_training_settings(pilot)
    run_id = build_run_id(commit, run_config.config_id, run_config.seed)

    state_path = Path(checkpoint_dir) / run_id / STATE_FILENAME
    fingerprint = _settings_fingerprint(fair)
    if not resume and state_path.exists():
        raise FileExistsError(
            f"실행 {run_id}의 체크포인트가 이미 있습니다. 이어서 학습하려면 resume=True로 부르고, "
            "처음부터 다시 하려면 체크포인트 폴더를 지우세요(이전 실행과 섞이지 않도록 자동으로 덮어쓰지 않습니다)."
        )

    dataset = PatchPoolDataset(data_dir, pilot["patch_pool_id"])
    if dataset.scale != SCALE:
        raise ValueError(f"패치 풀의 배율(x{dataset.scale})이 모델 배율 x{SCALE}와 다릅니다: {dataset.patch_pool_id}")
    if len(dataset) != pilot["patch_pool_size"]:
        raise ValueError(
            f"패치 풀 크기({len(dataset)})가 patch_pool_size({pilot['patch_pool_size']})와 다릅니다: {dataset.patch_pool_id}"
        )
    sampler = SeededEpochPatchSampler(
        len(dataset), pilot["patches_per_epoch"], run_config.seed, augmentation=pilot["augmentation"]
    )

    set_model_seed(run_config.seed)
    torch_device = torch.device(device)
    model, loss_fn = build_model_and_loss(run_config.switches, fair)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot["learning_rate"])

    epochs_total = pilot["epochs"]
    epoch_losses: list[float] = []
    train_seconds = 0.0
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"실행 {run_id}의 재개용 체크포인트가 없습니다.")
        state = torch.load(state_path, map_location=torch_device, weights_only=True)
        if state["run_id"] != run_id:
            raise ValueError(f"체크포인트의 run_id({state['run_id']})가 이번 실행({run_id})과 다릅니다.")
        if state["settings_sha256"] != fingerprint:
            raise ValueError(f"실행 {run_id}를 시작한 뒤 학습 설정(공정 비교 조건)이 바뀌어 이어서 학습할 수 없습니다.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch_losses = list(state["epoch_losses"])
        train_seconds = float(state["train_seconds"])

    max_seconds = fair["fixed_rules"]["max_hours"] * 3600.0
    model.train()
    for epoch in range(len(epoch_losses), epochs_total):
        if train_seconds >= max_seconds:  # 3시간 상한은 안전장치다. 에폭 단위로, 다음 에폭을 시작하기 전에만 판정한다.
            break
        epoch_started = clock()
        loss_sum, patch_count = 0.0, 0
        for lr_batch, hr_batch in dataset.iter_epoch_batches(sampler, epoch, pilot["batch_size"]):
            lr, hr = _to_tensor(lr_batch, torch_device), _to_tensor(hr_batch, torch_device)
            loss = loss_fn(model.forward_raw(lr), hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(lr_batch)
            patch_count += len(lr_batch)
        epoch_losses.append(loss_sum / patch_count)
        train_seconds += clock() - epoch_started
        _atomic_save(
            {
                "run_id": run_id,
                "settings_sha256": fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_losses": epoch_losses,
                "train_seconds": train_seconds,
            },
            state_path,
        )
    train_minutes = train_seconds / 60.0
    epochs_done = len(epoch_losses)
    status = COMPLETE if epochs_done == epochs_total else INCOMPLETE

    _atomic_save(model.state_dict(), Path(checkpoint_dir) / run_id / WEIGHTS_FILENAME)
    params = count_parameters(model)
    switches = run_config.switches
    _write_train_log(
        Path(output_dir) / run_id / TRAIN_LOG_FILENAME,
        {
            "run_id": run_id,
            "commit": commit,
            "config_id": run_config.config_id,
            "seed": run_config.seed,
            "patch_pool_id": pilot["patch_pool_id"],
            "switches": asdict(switches),
            "params": params,
            "status": status,
            "epochs_done": epochs_done,
            "epochs_total": epochs_total,
            "train_minutes": train_minutes,
            "epoch_losses": epoch_losses,
        },
    )
    return TrainResult(
        run_id=run_id,
        status=status,
        epochs_done=epochs_done,
        epochs_total=epochs_total,
        epoch_losses=epoch_losses,
        train_minutes=train_minutes,
        params=params,
        model=model,
    )


def train_and_evaluate(
    run_config: ResolvedRunConfig,
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    resume_run_id: str | None = None,
    allow_dirty: bool = False,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    device: str = "cpu",
) -> Path:
    """학습부터 통합 평가까지 실행 1회를 완결하고 실행 기록 폴더(`<output_dir>/<run_id>`)를 돌려줍니다.

    순서: 코드 커밋 확인(커밋 안 된 변경이면 거부, `allow_dirty`면 `-dirty` 표시) → 학습(학습 없는 구성은 건너뜀) →
    평가 체크포인트(마지막으로 끝낸 에폭의 모델)로 `evaluate_model`. 3시간 상한에 걸려 멈춘 실행도 평가하되
    `run_meta["status"]`를 `incomplete`로 넘긴다. `resume_run_id`가 있으면 그 실행을 이어서 학습하며,
    이번 실행의 run_id와 다르면 거부한다.
    """
    commit = check_git_clean(repo_root, allow_dirty=allow_dirty)
    run_id = build_run_id(commit, run_config.config_id, run_config.seed)
    if resume_run_id is not None and resume_run_id != run_id:
        raise ValueError(f"resume_run_id({resume_run_id})가 이번 실행의 run_id({run_id})와 다릅니다.")

    switches = run_config.switches
    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "commit": commit,
        "config_id": run_config.config_id,
        "seed": run_config.seed,
        "patch_pool_id": run_config.fair_comparison["pilot_determined"]["patch_pool_id"],
        "switches": asdict(switches),
    }
    model: torch.nn.Module | None = None
    if run_config.trainable:
        result = train_run(
            run_config,
            data_dir,
            output_dir,
            checkpoint_dir,
            commit=commit,
            resume=resume_run_id is not None,
            device=device,
        )
        model = result.model.eval()
        run_meta.update(params=result.params, train_minutes=result.train_minutes, status=result.status)
    else:
        run_meta.update(params=0, train_minutes=0.0, status=COMPLETE)

    run_dir = Path(output_dir) / run_id
    evaluate_model(model, run_meta, data_dir, run_dir)
    return run_dir
