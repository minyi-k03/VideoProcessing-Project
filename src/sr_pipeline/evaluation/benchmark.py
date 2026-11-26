"""파라미터 수 및 고정 입력 크기 추론 시간 측정 모듈 (SPEC.md / ADR-0003).

측정 조건(입력 크기, 워밍업·반복 횟수, 배율)은 `configs/fair_comparison.yaml`의 `infer_benchmark`·
`fixed_rules.scale`에서만 받고 이 모듈에는 기본값을 두지 않습니다. 배치 1, FP32, 워밍업 후 반복 평균,
GPU 동기화 포함. "네트워크만(infer_ms_net)"과 "색 변환·bicubic 포함 전체(infer_ms_total)"를 분리 기록합니다.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn

from sr_pipeline.common.bicubic import bicubic_resize


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _check_count(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}은(는) {minimum} 이상의 정수여야 합니다. 받은 값: {value!r}")


def _time_ms(fn: Callable[[], object], device: torch.device, warmup_runs: int, repeat_runs: int) -> float:
    """워밍업 후 `repeat_runs`번 돌린 전체 시간을 장치 동기화로 감싸 한 번당 평균 밀리초로 돌려준다."""
    for _ in range(warmup_runs):
        fn()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat_runs):
        fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeat_runs


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_time(
    model: nn.Module | None,
    *,
    scale: int,
    lr_height: int,
    lr_width: int,
    warmup_runs: int,
    repeat_runs: int,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """`(infer_ms_net, infer_ms_total)`을 밀리초 단위로 측정합니다.

    `scale`은 `fixed_rules.scale`, 나머지 측정 조건은 `infer_benchmark` 값을 호출자가 그대로 넘긴다.
    입력은 배치 1·FP32의 `(1, 3, lr_height, lr_width)` 난수 RGB다(값은 시간에 영향이 없다).
    - `infer_ms_total`: 색 변환·bicubic 경로를 포함한 `model(lr)` 전체. `model`이 `None`(Bicubic 하한)이면
      `scale`배 bicubic 확대(`common.bicubic.bicubic_resize`)만 잰다.
    - `infer_ms_net`: 신경망(`model._network`)만. 입력 채널은 구성에 맞춘다(RGB 3, Y 1). 신경망이 없는
      Bicubic 하한은 0.0이다.
    모델은 평가 모드로 두고 `torch.inference_mode()` 안에서 잰다. CPU 값은 T4 기준 값과 비교할 수 없다.
    """
    _check_count("scale", scale, 1)
    _check_count("lr_height", lr_height, 1)
    _check_count("lr_width", lr_width, 1)
    _check_count("warmup_runs", warmup_runs, 0)
    _check_count("repeat_runs", repeat_runs, 1)
    target = torch.device(device)
    lr = torch.rand(1, 3, lr_height, lr_width, dtype=torch.float32, device=target)

    if model is None:
        with torch.inference_mode():
            total_ms = _time_ms(lambda: bicubic_resize(lr, float(scale)), target, warmup_runs, repeat_runs)
        return 0.0, total_ms

    if not callable(getattr(model, "_network", None)):
        raise ValueError("네트워크만의 시간을 재려면 모델에 `_network`(신경망 단계)가 있어야 합니다.")
    was_training = model.training
    model.to(target).eval()
    try:
        in_channels = 1 if getattr(model, "use_y_channel", False) else 3
        net_input = torch.rand(1, in_channels, lr_height, lr_width, dtype=torch.float32, device=target)
        with torch.inference_mode():
            net_ms = _time_ms(lambda: model._network(net_input), target, warmup_runs, repeat_runs)
            total_ms = _time_ms(lambda: model(lr), target, warmup_runs, repeat_runs)
    finally:
        model.train(was_training)
    return net_ms, total_ms
