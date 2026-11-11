"""색·보간 공용 규칙 경계 테스트 (SPEC.md Testing Decisions §2)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Skeleton test: implement alongside common.color")
def test_bt601_y_matches_matlab_reference() -> None:
    """BT.601 Y 변환이 MATLAB/BasicSR 참조 구현과 허용 오차 안에서 일치해야 합니다."""


@pytest.mark.skip(reason="Skeleton test: implement alongside common.bicubic")
def test_bicubic_down_and_up_match_matlab_reference() -> None:
    """bicubic 축소(안티에일리어싱)와 확대(화소 중심 정렬)가 MATLAB 호환 참조 구현과 일치해야 합니다."""
