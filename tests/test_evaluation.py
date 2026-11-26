"""통합 평가 모듈 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sr_pipeline.diagnosis.diagnostics import compute_edge_flat_psnr
from sr_pipeline.evaluation.metrics import compute_edge_psnr, compute_lpips, compute_psnr, compute_ssim_y
from sr_pipeline.evaluation.postprocess import postprocess_to_y, quantize_and_shave


def _basicsr_ssim_reference(y1: np.ndarray, y2: np.ndarray) -> float:
    """BasicSR/MATLAB ssim_index(11x11 가우시안 sigma=1.5, valid 합성곱)를 numpy로 직접 옮긴 참조."""
    coords = np.arange(-5, 6, dtype=np.float64)
    g1d = np.exp(-(coords**2) / (2.0 * 1.5**2))
    g1d /= g1d.sum()
    window = np.outer(g1d, g1d)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    h, w = y1.shape
    out_h, out_w = h - 10, w - 10
    ssim_sum = 0.0
    for row in range(out_h):
        for col in range(out_w):
            p1 = y1[row : row + 11, col : col + 11]
            p2 = y2[row : row + 11, col : col + 11]
            mu1 = float(np.sum(window * p1))
            mu2 = float(np.sum(window * p2))
            var1 = float(np.sum(window * (p1 - mu1) ** 2))
            var2 = float(np.sum(window * (p2 - mu2) ** 2))
            cov12 = float(np.sum(window * (p1 - mu1) * (p2 - mu2)))
            num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov12 + c2)
            den = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
            ssim_sum += num / den
    return ssim_sum / (out_h * out_w)


def test_identical_images_psnr_inf_and_ssim_one() -> None:
    """같은 영상이면 PSNR(Y)은 무한대, SSIM(Y)은 1이어야 합니다."""
    generator = torch.Generator().manual_seed(0)
    textured = torch.rand(1, 3, 24, 24, generator=generator)
    flat = torch.full((1, 3, 24, 24), 0.5)

    for rgb in (textured, flat):
        y_sr = postprocess_to_y(rgb.clone(), border=4)
        y_hr = postprocess_to_y(rgb, border=4)
        assert compute_psnr(y_sr, y_hr) == math.inf
        assert compute_ssim_y(y_sr, y_hr) == 1.0


def test_border_4px_difference_does_not_affect_metrics() -> None:
    """가장자리 4화소 안쪽의 차이는 평가 지표에 영향을 주지 않아야 합니다."""
    generator = torch.Generator().manual_seed(1)
    hr = torch.rand(1, 3, 24, 28, generator=generator)
    # 내부에 작은 오차가 있는 SR과, 거기에 가장자리 4화소를 크게 훼손한 SR을 비교한다.
    sr_clean = hr.clone()
    sr_clean[:, :, 6:12, 8:14] += 0.05

    sr_dirty = sr_clean.clone()
    sr_dirty[:, :, :4, :] = -5.0
    sr_dirty[:, :, -4:, :] = 10.0
    sr_dirty[:, :, :, :4] = 0.0
    sr_dirty[:, :, :, -4:] = 1.0

    y_hr = postprocess_to_y(hr, border=4)
    y_clean = postprocess_to_y(sr_clean, border=4)
    y_dirty = postprocess_to_y(sr_dirty, border=4)

    psnr_clean = compute_psnr(y_clean, y_hr)
    ssim_clean = compute_ssim_y(y_clean, y_hr)
    assert math.isfinite(psnr_clean) and ssim_clean < 1.0
    assert compute_psnr(y_dirty, y_hr) == psnr_clean
    assert compute_ssim_y(y_dirty, y_hr) == ssim_clean

    # 내부까지 동일한 경우 가장자리 4화소 차이가 있어도 PSNR(Y)=inf, SSIM(Y)=1이어야 한다.
    sr_border_only = hr.clone()
    sr_border_only[:, :, :4, :] = 0.0
    y_border_only = postprocess_to_y(sr_border_only, border=4)
    assert compute_psnr(y_border_only, y_hr) == math.inf
    assert compute_ssim_y(y_border_only, y_hr) == 1.0


def test_edge_psnr_defined_on_completely_flat_image() -> None:
    """완전히 평탄한 영상에서도 에지 PSNR이 정의되어야 합니다."""
    hr_y = torch.full((1, 1, 16, 20), 100.0)
    assert compute_edge_psnr(hr_y.clone(), hr_y, top_q_percent=10.0) == math.inf
    # 평탄 HR에 전 화소 오차 1.0이면 에지 영역 선택과 무관하게 MSE = 1.
    assert compute_edge_psnr(hr_y + 1.0, hr_y, top_q_percent=10.0) == pytest.approx(
        20.0 * math.log10(255.0), abs=1e-12
    )


def test_edge_psnr_uses_only_top_q_percent_sobel_pixels_of_hr() -> None:
    """에지 PSNR은 HR Y Sobel 크기 상위 q% 화소의 오차만 반영합니다."""
    # 왼쪽 절반 0, 오른쪽 절반 100인 수직 경계 HR: Sobel 응답은 경계 두 열에만 있다.
    hr_y = torch.zeros(1, 1, 10, 10)
    hr_y[..., 5:] = 100.0
    edge_cols = [4, 5]
    sr_y = hr_y.clone()
    sr_y[..., edge_cols] += 3.0  # 에지 화소에만 오차 3.0
    # 상위 20% = 200 / 10행 x 2열 = 20화소 -> 정확히 경계 두 열 -> MSE 9
    assert compute_edge_psnr(sr_y, hr_y, top_q_percent=20.0) == pytest.approx(
        10.0 * math.log10(255.0**2 / 9.0), abs=1e-9
    )
    # 에지가 아닌 곳의 큰 오차는 영향이 없다.
    sr_far = hr_y.clone()
    sr_far[..., :2] += 50.0
    assert compute_edge_psnr(sr_far, hr_y, top_q_percent=20.0) == math.inf


def test_edge_psnr_matches_diagnosis_edge_region() -> None:
    """평가의 에지 영역 정의가 진단 모듈과 갈라지지 않아야 합니다."""
    generator = torch.Generator().manual_seed(3)
    hr_y = torch.rand(1, 1, 24, 30, generator=generator) * 219.0 + 16.0
    sr_y = hr_y + 5.0 * torch.randn(1, 1, 24, 30, generator=generator)
    expected = compute_edge_flat_psnr(sr_y, hr_y, 15.0)["edge_psnr_y"]
    assert compute_edge_psnr(sr_y, hr_y, top_q_percent=15.0) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("q", [0.0, 100.0, -5.0, 150.0, float("nan")])
def test_edge_psnr_rejects_invalid_q(q: float) -> None:
    y = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError):
        compute_edge_psnr(y, y, top_q_percent=q)


def test_clamp_and_quantize_order_consistent_across_configs() -> None:
    """자르기([0,1]) -> 8비트 양자화 -> 가장자리 4화소 잘라내기 순서가 일관되어야 합니다."""
    # 범위 밖 값(-0.2, 1.4)과 중간 소수값(100.4/255, 100.6/255)을 포함한 (1, 3, 12, 14) 영상.
    # uint8 변환 전에 [0, 1]로 자르지 않으면 -0.2 * 255가 underflow/wraparound된다.
    img = torch.full((1, 3, 12, 14), 100.4 / 255.0, dtype=torch.float32)
    img[:, :, 5, 5] = -0.2
    img[:, :, 5, 6] = 1.4
    img[:, :, 6, 5] = 100.6 / 255.0
    # 가장자리 4화소 안에는 다른 값을 두어 shave로 제거되는지 확인한다.
    img[:, :, 0, 0] = 0.9
    img[:, :, -1, -1] = 0.1

    out = quantize_and_shave(img, border=4)

    assert out.shape == (1, 3, 4, 6)
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    # 반환값은 정확히 8비트(k / 255) 이산 값이어야 한다.
    levels = out * 255.0
    assert torch.allclose(levels, levels.round(), atol=1e-5)
    # 내부 좌표 (5, 5) -> shave 후 (1, 1)는 0, (5, 6) -> (1, 2)는 255, (6, 5) -> (2, 1)는 101
    assert out[0, 0, 1, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[0, 0, 1, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert out[0, 0, 2, 1].item() == pytest.approx(101.0 / 255.0, abs=1e-6)
    assert out[0, 0, 0, 0].item() == pytest.approx(100.0 / 255.0, abs=1e-6)


@pytest.mark.parametrize(
    ("img", "border"),
    [
        (torch.rand(3, 12, 12), 4),  # 4D 아님
        (torch.rand(1, 0, 12, 12), 4),  # 빈 채널
        (torch.randint(0, 256, (1, 3, 12, 12), dtype=torch.uint8), 4),  # float 아님
        (torch.rand(1, 3, 12, 12), -1),  # 음수 border
        (torch.rand(1, 3, 8, 12), 4),  # border 잘라낸 뒤 높이 0
        (torch.tensor([[[[float("nan")] * 10] * 10] * 3]), 1),  # NaN 입력
        (torch.tensor([[[[float("inf")] * 10] * 10] * 3]), 1),  # Inf 입력
    ],
)
def test_quantize_and_shave_rejects_invalid_input(img: torch.Tensor, border: int) -> None:
    with pytest.raises(ValueError):
        quantize_and_shave(img, border=border)


def test_postprocess_requires_explicit_border() -> None:
    """border는 fair_comparison.yaml 값이라 기본값 없이 호출하는 쪽이 넘겨야 합니다."""
    sample = torch.rand(1, 3, 16, 16)
    with pytest.raises(TypeError):
        quantize_and_shave(sample)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        postprocess_to_y(sample)  # type: ignore[call-arg]


def test_compute_psnr_matches_known_values() -> None:
    """알려진 오차에 대한 PSNR이 정의(10·log10(max_val²/MSE))와 일치해야 합니다."""
    hr_y = torch.full((1, 1, 4, 5), 100.0, dtype=torch.float64)
    assert compute_psnr(hr_y.clone(), hr_y) == math.inf

    # 전 화소 오차 1.0 -> MSE = 1.0 -> PSNR = 20·log10(255)
    assert compute_psnr(hr_y + 1.0, hr_y) == pytest.approx(20.0 * math.log10(255.0), abs=1e-12)

    # 6화소 중 1화소만 3.0 차이 -> MSE = 9 / 6 = 1.5
    small_hr = torch.full((1, 1, 2, 3), 50.0, dtype=torch.float64)
    small_sr = small_hr.clone()
    small_sr[0, 0, 1, 2] = 53.0
    expected = 10.0 * math.log10((255.0**2) / 1.5)
    assert compute_psnr(small_sr, small_hr) == pytest.approx(expected, abs=1e-12)

    # max_val=1.0 [0, 1] 범위와 max_val=255.0 [0, 255] 범위가 같은 PSNR을 낸다.
    assert compute_psnr(small_sr / 255.0, small_hr / 255.0, max_val=1.0) == pytest.approx(
        expected, abs=1e-10
    )


@pytest.mark.parametrize(
    ("pred", "target", "max_val"),
    [
        (torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), 255.0),  # batch != 1
        (torch.rand(1, 8, 8), torch.rand(1, 8, 8), 255.0),  # 4D 아님
        (torch.randint(0, 256, (1, 1, 8, 8), dtype=torch.uint8), torch.rand(1, 1, 8, 8), 255.0),
        (torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9), 255.0),  # shape 불일치
        (torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), 0.0),  # max_val <= 0
        (torch.full((1, 1, 8, 8), float("nan")), torch.rand(1, 1, 8, 8), 255.0),
    ],
)
def test_compute_psnr_rejects_invalid_inputs(
    pred: torch.Tensor, target: torch.Tensor, max_val: float
) -> None:
    with pytest.raises(ValueError):
        compute_psnr(pred, target, max_val=max_val)


def test_compute_ssim_y_matches_analytical_constant_pair() -> None:
    """상수 영상 쌍에서는 가우시안 창과 무관하게 SSIM(Y) = (2μ₁μ₂ + C₁) / (μ₁² + μ₂² + C₁)이다."""
    y1 = torch.full((1, 1, 16, 16), 100.0, dtype=torch.float32)
    y2 = torch.full((1, 1, 16, 16), 110.0, dtype=torch.float32)
    c1 = (0.01 * 255.0) ** 2
    expected = (2.0 * 100.0 * 110.0 + c1) / (100.0**2 + 110.0**2 + c1)
    assert compute_ssim_y(y1, y2) == pytest.approx(expected, abs=1e-12)


def test_compute_ssim_y_matches_matlab_basicsr_reference() -> None:
    """SSIM(Y)가 BasicSR/MATLAB ssim_index 참조 구현과 허용 오차 안에서 일치해야 합니다."""
    generator = torch.Generator().manual_seed(7)
    hr_y = torch.rand(1, 1, 23, 27, generator=generator, dtype=torch.float32) * 219.0 + 16.0
    sr_y = (hr_y + 12.0 * torch.randn(1, 1, 23, 27, generator=generator)).clamp(16.0, 235.0)

    expected = _basicsr_ssim_reference(
        sr_y[0, 0].double().numpy(),
        hr_y[0, 0].double().numpy(),
    )
    assert compute_ssim_y(sr_y, hr_y) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize(
    ("sr_y", "hr_y"),
    [
        (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)),  # 1채널 Y가 아님
        (torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)),  # batch != 1
        (torch.rand(1, 1, 10, 16), torch.rand(1, 1, 10, 16)),  # 11x11 창보다 작음
        (torch.rand(1, 1, 16, 10), torch.rand(1, 1, 16, 10)),  # 11x11 창보다 작음
        (torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 17)),  # shape 불일치
        (torch.randint(16, 235, (1, 1, 16, 16), dtype=torch.uint8), torch.rand(1, 1, 16, 16)),
    ],
)
def test_compute_ssim_y_rejects_invalid_inputs(sr_y: torch.Tensor, hr_y: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        compute_ssim_y(sr_y, hr_y)


class _RecordingLpips(torch.nn.Module):
    """입력을 기록하고 입력 차이의 평균 제곱을 돌려주는 LPIPS 대역(실제 AlexNet 가중치 불필요)."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.inputs.append((x.clone(), y.clone()))
        return (x - y).pow(2).mean().view(1, 1, 1, 1)


def test_compute_lpips_feeds_minus_one_to_one_rgb_and_returns_float() -> None:
    """[0, 1] RGB가 [-1, 1]로 바뀌어 LPIPS 모델에 들어가고, 결과는 파이썬 float여야 합니다."""
    sr = torch.zeros(1, 3, 12, 12)
    hr = torch.ones(1, 3, 12, 12)
    stub = _RecordingLpips()

    value = compute_lpips(sr, hr, lpips_model=stub)

    (fed_sr, fed_hr), = stub.inputs
    assert fed_sr.min().item() == -1.0 and fed_sr.max().item() == -1.0
    assert fed_hr.min().item() == 1.0 and fed_hr.max().item() == 1.0
    assert isinstance(value, float)
    assert value == pytest.approx(4.0)  # (-1 - 1)^2


@pytest.mark.parametrize(
    ("sr", "hr"),
    [
        (torch.rand(1, 1, 12, 12), torch.rand(1, 1, 12, 12)),  # RGB 아님
        (torch.rand(2, 3, 12, 12), torch.rand(2, 3, 12, 12)),  # batch != 1
        (torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 13)),  # shape 불일치
    ],
)
def test_compute_lpips_rejects_invalid_inputs(sr: torch.Tensor, hr: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        compute_lpips(sr, hr, lpips_model=_RecordingLpips())


def test_compute_lpips_real_alexnet_identical_zero_and_different_positive() -> None:
    """실제 AlexNet LPIPS(가중치 다운로드 필요): 같은 영상은 0, 다른 영상은 양수. 가중치를 못 받으면 건너뜁니다."""
    generator = torch.Generator().manual_seed(5)
    a = torch.rand(1, 3, 64, 64, generator=generator)
    b = torch.rand(1, 3, 64, 64, generator=generator)
    try:
        same = compute_lpips(a, a)
    except Exception as error:  # 오프라인 등으로 백본 가중치를 못 받는 경우
        pytest.skip(f"AlexNet LPIPS를 만들 수 없습니다: {error}")
    assert same == pytest.approx(0.0, abs=1e-6)
    assert compute_lpips(a, b) > 0.0
