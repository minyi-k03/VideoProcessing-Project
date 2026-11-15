"""로컬 CPU 테스트용 공용 합성 데이터 픽스처 (SPEC.md Testing Decisions)."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch


@pytest.fixture
def restore_global_random_state() -> Iterator[None]:
    """set_model_seed가 바꾸는 프로세스 전역 상태(random·numpy·torch CPU/CUDA 난수, cuDNN 플래그)를
    테스트가 끝나면 되돌려 다른 테스트에 새지 않게 한다. 쓰는 모듈은 `pytestmark = usefixtures(...)`로 켠다."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags


@pytest.fixture
def synthetic_image_pair():
    """GPU 없이 테스트할 수 있는 작은 합성 (LR, HR) 텐서 쌍 스텁."""
    pytest.skip("Skeleton fixture: provide small deterministic RGB LR/HR tensors once implemented.")
