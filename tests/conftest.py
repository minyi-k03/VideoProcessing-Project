"""로컬 CPU 테스트용 공용 합성 데이터 픽스처 (SPEC.md Testing Decisions)."""

from __future__ import annotations

import pytest


@pytest.fixture
def synthetic_image_pair():
    """GPU 없이 테스트할 수 있는 작은 합성 (LR, HR) 텐서 쌍 스텁."""
    pytest.skip("Skeleton fixture: provide small deterministic RGB LR/HR tensors once implemented.")
