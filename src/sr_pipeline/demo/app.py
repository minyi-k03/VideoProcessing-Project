"""라이브 데모 실행기 (SPEC.md / User Stories 79~81).

VS Code Colab T4 연결(또는 GPU 미배정 시 CPU 폴백)에서 저장된 체크포인트를 불러와
Bicubic 하한 / 베이스라인 / 제안 구성 3분할 비교와 고주파 ROI 확대를 표시합니다.
"""

from __future__ import annotations

from pathlib import Path


def build_three_way_comparison(
    lr_image_path: Path,
    baseline_ckpt: Path,
    proposed_ckpt: Path,
    proposed_config_id: str,
    device: str = "cuda",
) -> Path:
    """Bicubic 하한 / 베이스라인 / 제안 구성 3분할 비교 이미지를 생성합니다."""
    raise NotImplementedError("TODO: Run inference on CPU/GPU and render 3-way split + zoomed ROI.")


def launch_demo(
    baseline_ckpt: Path,
    proposed_ckpt: Path,
    proposed_config_id: str,
    sample_dir: Path,
    output_dir: Path,
) -> None:
    """대표 영상 및 선택적 업로드 영상 대상 라이브 데모를 실행합니다."""
    raise NotImplementedError("TODO: Launch interactive or script-driven live demo.")
