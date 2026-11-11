"""파라미터 수 및 고정 입력 크기 추론 시간 측정 모듈 (SPEC.md / ADR-0003).

측정 조건: LR 320x180 -> SR 1280x720, T4 GPU, 배치 1, FP32, 워밍업 10회 후 100회 평균,
GPU 동기화 포함. "네트워크만(infer_ms_net)"과 "색 변환·bicubic 포함 전체(infer_ms_total)"를 분리 기록합니다.
"""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: nn.Module,
    lr_height: int = 180,
    lr_width: int = 320,
    warmup_runs: int = 10,
    repeat_runs: int = 100,
) -> tuple[float, float]:
    """`(infer_ms_net, infer_ms_total)`을 밀리초 단위로 측정합니다."""
    raise NotImplementedError("TODO: Benchmark net-only and end-to-end GPU inference latency.")
