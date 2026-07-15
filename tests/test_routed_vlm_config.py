"""Tests for routed VLM serving configuration helpers."""

from __future__ import annotations

from src.serving.routed_vlm import RoutedVLMService


class FakeTorch:
    float16 = "float16"
    float32 = "float32"


def test_torch_dtype_for_accelerators_uses_float16() -> None:
    assert RoutedVLMService._torch_dtype_for_device(FakeTorch, "cuda") == "float16"
    assert RoutedVLMService._torch_dtype_for_device(FakeTorch, "mps") == "float16"


def test_torch_dtype_for_cpu_uses_float32() -> None:
    assert RoutedVLMService._torch_dtype_for_device(FakeTorch, "cpu") == "float32"
