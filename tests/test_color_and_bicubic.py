"""색·보간 공용 규칙 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import sr_pipeline.common.bicubic as bicubic_module
import sr_pipeline.common.color as color_module
from sr_pipeline.common.bicubic import bicubic_resize
from sr_pipeline.common.color import (
    denormalize_y,
    normalize_y,
    rgb_to_y,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def _matlab_rgb2ycbcr_reference(rgb: np.ndarray) -> np.ndarray:
    """BasicSR/MATLAB rgb2ycbcr(uint8 스케일)의 정의를 numpy로 직접 옮긴 참조 구현. rgb: (H, W, 3).

    구현과 같은 계수를 다시 적은 것이라 계수 오타는 못 잡는다. 계수 자체는 아래
    test_bt601_primary_colors_known_values의 널리 알려진 고정 값이 독립적으로 지킨다.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128.0 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128.0 + 112.0 * r - 93.786 * g - 18.214 * b
    return np.stack([y, cb, cr], axis=-1)


def _random_rgb(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, 8, 9, generator=generator, dtype=torch.float64)


@pytest.fixture
def fresh_color_cache():
    """상수 캐시 상태가 테스트끼리 새지 않게 앞뒤로 비운다."""
    color_module._constants.cache_clear()
    yield
    color_module._constants.cache_clear()


def test_bt601_y_matches_matlab_reference() -> None:
    """BT.601 Y 변환이 MATLAB/BasicSR 참조 구현과 허용 오차 안에서 일치해야 합니다(배치 전체)."""
    rgb = _random_rgb()
    ycbcr = rgb_to_ycbcr(rgb)
    y_only = rgb_to_y(rgb)

    for index in range(rgb.shape[0]):
        expected = _matlab_rgb2ycbcr_reference(rgb[index].permute(1, 2, 0).numpy())
        np.testing.assert_allclose(ycbcr[index].permute(1, 2, 0).numpy(), expected, atol=1e-9)
        np.testing.assert_allclose(y_only[index, 0].numpy(), expected[..., 0], atol=1e-9)


@pytest.mark.parametrize(
    ("rgb", "expected_ycbcr"),
    [
        ((1.0, 0.0, 0.0), (81.481, 90.203, 240.0)),
        ((0.0, 1.0, 0.0), (144.553, 53.797, 34.214)),
        ((0.0, 0.0, 1.0), (40.966, 240.0, 109.786)),
        ((1.0, 1.0, 1.0), (235.0, 128.0, 128.0)),
        ((0.0, 0.0, 0.0), (16.0, 128.0, 128.0)),
    ],
)
def test_bt601_primary_colors_known_values(rgb, expected_ycbcr) -> None:
    """원색·흑백의 BT.601 값(적색 81.481/90.203/240 등)이 알려진 고정 값과 일치해야 합니다.

    구현 계수를 다시 쓰지 않은 값이라, 계수 오타가 참조 구현과 같이 틀려도 여기서 잡힌다.
    rgb_to_y도 같은 값(Y)을 내야 한다.
    """
    pixel = torch.tensor(rgb, dtype=torch.float64).view(1, 3, 1, 1)
    assert rgb_to_ycbcr(pixel).flatten().tolist() == pytest.approx(expected_ycbcr, abs=1e-3)
    assert rgb_to_y(pixel).item() == pytest.approx(expected_ycbcr[0], abs=1e-3)


def test_bt601_range_endpoints() -> None:
    """검정은 Y=16, 흰색은 Y=235, 회색의 Cb·Cr은 128이어야 합니다."""
    black = torch.zeros(1, 3, 2, 2)
    white = torch.ones(1, 3, 2, 2)
    gray = torch.full((1, 3, 2, 2), 0.5)

    assert torch.allclose(rgb_to_y(black), torch.full((1, 1, 2, 2), 16.0))
    assert torch.allclose(rgb_to_y(white), torch.full((1, 1, 2, 2), 235.0), atol=1e-4)
    gray_ycbcr = rgb_to_ycbcr(gray)
    assert torch.allclose(gray_ycbcr[:, 1:], torch.full((1, 2, 2, 2), 128.0), atol=1e-4)


def test_rgb_ycbcr_round_trip_does_not_clip() -> None:
    """RGB→YCbCr→RGB 왕복이 원래 값을 복원해야 하고, [0, 1] 밖의 값도 자르지 않아야 합니다."""
    rgb = _random_rgb(1)
    assert torch.allclose(ycbcr_to_rgb(rgb_to_ycbcr(rgb)), rgb, atol=1e-9)

    out_of_range = rgb * 1.5 - 0.25
    restored = ycbcr_to_rgb(rgb_to_ycbcr(out_of_range))
    assert torch.allclose(restored, out_of_range, atol=1e-9)
    assert restored.min() < 0.0 and restored.max() > 1.0


def test_normalize_y_round_trip_and_range() -> None:
    """Y 정규화는 16→0, 235→1로 보내고, 되돌리면 원래 값이어야 합니다."""
    y = torch.tensor([16.0, 125.5, 235.0]).view(1, 1, 1, 3)
    normalized = normalize_y(y)
    assert torch.allclose(normalized[..., 0], torch.tensor(0.0))
    assert torch.allclose(normalized[..., 2], torch.tensor(1.0))
    assert torch.allclose(denormalize_y(normalized), y)


def test_normalize_y_does_not_clip_out_of_range_values() -> None:
    """자르기 전 출력(Y가 16~235 밖)도 정규화가 자르지 않고 [0, 1] 밖 값으로 보존해야 합니다."""
    y = torch.tensor([0.0, 300.0]).view(1, 1, 1, 2)
    normalized = normalize_y(y)
    assert normalized[..., 0] < 0.0 and normalized[..., 1] > 1.0
    assert torch.allclose(denormalize_y(normalized), y, atol=1e-4)  # float32 반올림 오차 허용


def test_color_functions_reject_wrong_channels() -> None:
    """3채널이 아닌 입력은 조용히 넘어가지 않고 오류여야 합니다."""
    with pytest.raises(ValueError):
        rgb_to_ycbcr(torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        rgb_to_y(torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        ycbcr_to_rgb(torch.zeros(3, 4, 4))


def test_color_float32_keeps_dtype_and_accuracy() -> None:
    """본 실행 정밀도(FP32)에서 모든 함수가 dtype을 유지하고 왕복 오차가 작아야 합니다."""
    rgb = _random_rgb(2).float()
    assert rgb_to_ycbcr(rgb).dtype == torch.float32
    assert rgb_to_y(rgb).dtype == torch.float32
    assert normalize_y(rgb_to_y(rgb)).dtype == torch.float32
    assert denormalize_y(normalize_y(rgb_to_y(rgb))).dtype == torch.float32
    restored = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    assert restored.dtype == torch.float32
    assert torch.allclose(restored, rgb, atol=1e-4)


def test_color_gradients_flow_with_expected_values(fresh_color_cache) -> None:
    """B·B′는 손실 그래프 안에서 이 함수들을 부르므로 역전파가 되어야 하고 기울기가 계수와 같아야 합니다."""
    rgb = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    rgb_to_y(rgb).sum().backward()
    expected = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float64).view(1, 3, 1, 1)
    assert torch.allclose(rgb.grad, expected.expand_as(rgb))

    ycbcr = torch.rand(1, 3, 4, 4, requires_grad=True)
    ycbcr_to_rgb(ycbcr).sum().backward()
    assert ycbcr.grad is not None and torch.isfinite(ycbcr.grad).all()


def test_backward_works_after_constants_first_built_in_inference_mode(fresh_color_cache) -> None:
    """평가·추론 시간 측정이 inference_mode로 먼저 돌아도, 이후 학습 역전파가 깨지면 안 됩니다."""
    with torch.inference_mode():
        sample = torch.rand(1, 3, 4, 4)
        rgb_to_ycbcr(sample)
        rgb_to_y(sample)
        ycbcr_to_rgb(sample)

    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    (rgb_to_ycbcr(rgb).sum() + rgb_to_y(rgb).sum() + ycbcr_to_rgb(rgb).sum()).backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()


def _cubic_kernel_reference(x: np.ndarray) -> np.ndarray:
    """Keys cubic 커널(a = -0.5). MATLAB imresize 'bicubic' 커널의 정의."""
    absx = np.abs(x)
    absx2, absx3 = absx**2, absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((absx > 1) & (absx <= 2))


def _resize_axis_reference(
    data: np.ndarray, axis: int, scale: float, antialiasing: bool
) -> np.ndarray:
    """BasicSR `matlab_functions.imresize`의 한 축 처리를 numpy 반복문으로 옮긴 참조 구현.

    구현(밀집 행렬 matmul)과 구조를 달리한다: 출력 화소마다 가중치·인덱스를 구하고, 입력을 대칭
    패딩한 뒤 화소별로 합산한다. 출력 크기는 ceil(in * scale), 좌표는 MATLAB의 1-indexed 규칙이다.
    """
    in_length = data.shape[axis]
    out_length = int(np.ceil(in_length * scale))
    shrink = scale < 1 and antialiasing
    kernel_width = 4.0 / scale if shrink else 4.0

    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    taps = int(np.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps)[None, :]
    distance = u[:, None] - indices
    if shrink:
        weights = scale * _cubic_kernel_reference(distance * scale)
    else:
        weights = _cubic_kernel_reference(distance)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # 첫 탭이 입력 0보다 오른쪽에서 시작하면(극단 축소) 앞쪽 패딩은 필요 없다. 패딩 길이와
    # 인덱스 오프셋이 같은 값을 쓰도록 0으로 자른다.
    pad_start = max(int(-indices.min() + 1), 0)
    pad_end = max(int(indices.max() - in_length), 0)
    moved = np.moveaxis(data, axis, 0)
    padded = np.pad(
        moved,
        [(pad_start, pad_end)] + [(0, 0)] * (moved.ndim - 1),
        mode="symmetric",
    )
    positions = (indices + pad_start - 1).astype(np.int64)  # 패딩된 배열 기준 0-indexed
    out = np.zeros((out_length,) + moved.shape[1:], dtype=np.float64)
    for row in range(out_length):
        for tap in range(taps):
            out[row] += weights[row, tap] * padded[positions[row, tap]]
    return np.moveaxis(out, 0, axis)


def _matlab_imresize_reference(
    image: np.ndarray, scale: float, antialiasing: bool = True
) -> np.ndarray:
    """(C, H, W) 영상을 H 방향 다음 W 방향으로 리사이즈하는 MATLAB 호환 참조 구현."""
    resized = _resize_axis_reference(image.astype(np.float64), 1, scale, antialiasing)
    return _resize_axis_reference(resized, 2, scale, antialiasing)


@pytest.fixture
def fresh_bicubic_cache():
    """가중치 캐시 상태가 테스트끼리 새지 않게 앞뒤로 비운다."""
    bicubic_module._weights.cache_clear()
    yield
    bicubic_module._weights.cache_clear()


@pytest.mark.parametrize(
    ("height", "width", "scale", "antialiasing"),
    [
        (32, 32, 0.25, True),  # 열화 모델: x4 축소 (안티에일리어싱)
        (32, 32, 0.25, False),
        (17, 23, 0.25, True),  # 4의 배수가 아닌 크기: 출력은 ceil
        (8, 8, 4.0, True),  # Cb·Cr 확대, 전역 잔차, Bicubic 하한: x4 확대
        (9, 14, 4.0, False),
        (11, 7, 2.0, True),  # 비정사각 확대
        (1, 1, 4.0, True),  # 커널 지지 범위(4화소)보다 작은 입력: 대칭 반사가 여러 번 접힌다
        (2, 3, 4.0, True),
        (3, 5, 0.25, True),
        (40, 40, 0.1, False),  # 안티에일리어싱 없이 극단 축소: 첫 탭이 입력 0보다 오른쪽에서 시작한다
    ],
)
@pytest.mark.parametrize(("dtype", "atol"), [(torch.float64, 1e-9), (torch.float32, 1e-5)])
def test_bicubic_down_and_up_match_matlab_reference(
    height, width, scale, antialiasing, dtype, atol
) -> None:
    """bicubic 축소(안티에일리어싱)와 확대(화소 중심 정렬)가 MATLAB 호환 참조 구현과 일치해야 합니다."""
    generator = torch.Generator().manual_seed(0)
    batch = torch.rand(2, 3, height, width, generator=generator, dtype=torch.float64)

    actual = bicubic_resize(batch.to(dtype), scale, antialiasing=antialiasing)

    for index in range(batch.shape[0]):
        expected = _matlab_imresize_reference(batch[index].numpy(), scale, antialiasing)
        np.testing.assert_allclose(actual[index].numpy(), expected, atol=atol)


@pytest.mark.parametrize(
    ("height", "width", "scale", "expected_shape"),
    [
        (32, 32, 0.25, (8, 8)),
        (17, 23, 0.25, (5, 6)),  # ceil(4.25), ceil(5.75)
        (9, 14, 4.0, (36, 56)),
        (11, 7, 2.0, (22, 14)),
    ],
)
def test_bicubic_output_size_is_ceil_of_input_times_scale(
    height, width, scale, expected_shape
) -> None:
    """출력 크기는 MATLAB과 같이 ceil(입력 크기 x 배율)이고, 배치·채널 수는 유지돼야 합니다."""
    out = bicubic_resize(torch.zeros(2, 3, height, width), scale)
    assert tuple(out.shape) == (2, 3, *expected_shape)


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_bicubic_preserves_constant_image(scale) -> None:
    """상수 영상은 축소·확대 후에도 같은 값이어야 합니다(가중치 합 1, 가장자리 포함)."""
    out = bicubic_resize(torch.full((1, 3, 16, 20), 0.37, dtype=torch.float64), scale)
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-12)


def test_bicubic_x4_upscale_keeps_linear_ramp_linear_in_interior() -> None:
    """x4 확대에서 선형 램프는 (가장자리 반사 영향이 없는) 내부에서 선형이어야 합니다.

    화소 중심 정렬이면 출력 화소 j(0부터)는 입력 좌표 j/4 - 0.375에 해당한다. 입력이 w(0부터)인
    램프이므로 기대값은 그 좌표 자체다. Keys cubic(a=-0.5)은 선형 함수를 정확히 재현한다.
    """
    ramp = torch.arange(16, dtype=torch.float64).view(1, 1, 1, 16).expand(1, 1, 4, 16)
    out = bicubic_resize(ramp, 4.0)

    j = torch.arange(8, 56, dtype=torch.float64)  # 가장자리에서 2화소 이상 떨어진 내부
    assert torch.allclose(out[0, 0, 1, 8:56], j / 4 - 0.375, atol=1e-9)


def test_bicubic_x4_upscale_matches_hand_computed_catmull_rom_value() -> None:
    """x4 확대의 한 출력 화소를 교과서 Catmull-Rom(Keys a=-0.5) 가중치로 손계산한 값과 비교합니다.

    위 참조 구현과 공식을 공유하지 않는 고정값이다. 출력 화소 j(0부터)는 입력 좌표 j/4 - 0.375에
    해당하므로 j = 19는 입력 4와 5 사이(t = 0.375)이고, 사용하는 입력은 v[3..6] = 1, 5, 9, 2다.
    t = 0.375의 가중치는 (-t^3+2t^2-t)/2, (3t^3-5t^2+2)/2, (-3t^3+4t^2+t)/2, (t^3-t^2)/2
    = -0.0732421875, 0.7275390625, 0.3896484375, -0.0439453125 (합 1)이다.
    기대값 = -0.0732421875*1 + 0.7275390625*5 + 0.3896484375*9 - 0.0439453125*2 = 6.9833984375.
    """
    values = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0])
    row = values.to(torch.float64).view(1, 1, 1, 12)
    out = bicubic_resize(row, 4.0)
    assert out[0, 0, 0, 19].item() == pytest.approx(6.9833984375, abs=1e-12)


def test_bicubic_scale_one_is_identity() -> None:
    """MATLAB imresize처럼 배율 1은 입력을 그대로 돌려줘야 합니다(보간 위치가 정확히 화소 중심)."""
    image = torch.rand(1, 3, 9, 11, dtype=torch.float64)
    assert torch.allclose(bicubic_resize(image, 1.0), image, atol=1e-12)


def test_bicubic_antialiasing_matters_for_downscale_only() -> None:
    """4화소 주기 세로선 x4 축소: 안티에일리어싱을 켜면 평균(0.25) 근처, 끄면 앨리어싱으로 벗어납니다.

    선이 col % 4 == 1에 있으면 x4 축소의 샘플 위치(입력 좌표 4x - 2.5, 0부터)가 선 바로 옆이라,
    끄면 선의 밝기만 과하게 뽑힌다(약 0.56). 확대에서는 옵션이 결과를 바꾸지 않아야 합니다.
    """
    cols = torch.arange(32).view(1, 1, 1, 32)
    lines = (cols % 4 == 1).to(torch.float64).expand(1, 1, 32, 32)

    with_aa = bicubic_resize(lines, 0.25, antialiasing=True)[..., 2:-2, 2:-2]
    without_aa = bicubic_resize(lines, 0.25, antialiasing=False)[..., 2:-2, 2:-2]
    assert (with_aa - 0.25).abs().max() < 0.05
    assert (without_aa - 0.25).abs().max() > 0.25

    small = torch.rand(1, 3, 6, 6, dtype=torch.float64)
    assert torch.equal(
        bicubic_resize(small, 4.0, antialiasing=True),
        bicubic_resize(small, 4.0, antialiasing=False),
    )


def test_bicubic_does_not_clip_or_round() -> None:
    """출력은 자르지도 반올림하지도 않아야 합니다(자르기·양자화는 postprocess 단계의 일)."""
    step = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    step[..., 4:] = 1.0
    out = bicubic_resize(step, 4.0)
    assert out.min() < 0.0 and out.max() > 1.0  # 계단 에지 오버슈트가 보존된다

    integer_valued = torch.randint(0, 256, (1, 1, 8, 8)).to(torch.float64)
    resized = bicubic_resize(integer_valued, 4.0)
    assert (resized != resized.round()).any()


def test_bicubic_batch_and_channels_are_independent() -> None:
    """(B, C)를 한 번에 처리한 결과가 영상·채널별로 따로 처리한 결과와 같아야 합니다."""
    generator = torch.Generator().manual_seed(3)
    batch = torch.rand(2, 3, 12, 10, generator=generator, dtype=torch.float64)
    together = bicubic_resize(batch, 0.5)
    for b in range(2):
        for c in range(3):
            alone = bicubic_resize(batch[b : b + 1, c : c + 1], 0.5)
            assert torch.allclose(together[b : b + 1, c : c + 1], alone, atol=1e-12)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_bicubic_gradients_are_exact_and_finite(scale, fresh_bicubic_cache) -> None:
    """C·B가 학습 그래프 안에서 이 함수를 부르므로 역전파가 되어야 하고 기울기가 정확해야 합니다."""
    x = torch.rand(1, 1, 5, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: bicubic_resize(t, scale), (x,))

    x32 = torch.rand(1, 3, 8, 8, requires_grad=True)
    bicubic_resize(x32, scale).sum().backward()
    assert x32.grad is not None and torch.isfinite(x32.grad).all()


def test_backward_works_after_weights_first_built_in_inference_mode(fresh_bicubic_cache) -> None:
    """평가·추론 시간 측정이 inference_mode로 먼저 돌아도, 이후 학습 역전파가 깨지면 안 됩니다."""
    with torch.inference_mode():
        sample = torch.rand(1, 3, 16, 16)
        bicubic_resize(sample, 0.25)
        bicubic_resize(sample, 4.0)

    x = torch.rand(1, 3, 16, 16, requires_grad=True)
    (bicubic_resize(x, 0.25).sum() + bicubic_resize(x, 4.0).sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("img", "scale"),
    [
        (torch.zeros(3, 8, 8), 0.5),  # (B, C, H, W)가 아님
        (torch.zeros(1, 3, 8, 8, dtype=torch.uint8), 0.5),  # 정수 dtype: 조용한 형 변환 금지
        (torch.zeros(1, 3, 8, 8), 0.0),
        (torch.zeros(1, 3, 8, 8), -2.0),
        (torch.zeros(1, 3, 8, 8), float("nan")),
        (torch.zeros(1, 3, 0, 8), 0.5),  # 빈 영상
    ],
)
def test_bicubic_rejects_invalid_input(img, scale) -> None:
    """잘못된 입력은 조용히 넘어가지 않고 오류여야 합니다."""
    with pytest.raises(ValueError):
        bicubic_resize(img, scale)


def test_bicubic_keeps_dtype() -> None:
    """본 실행 정밀도(FP32)와 float64 모두 dtype을 유지해야 합니다."""
    assert bicubic_resize(torch.rand(1, 3, 8, 8), 4.0).dtype == torch.float32
    assert bicubic_resize(torch.rand(1, 3, 8, 8, dtype=torch.float64), 0.5).dtype == torch.float64
