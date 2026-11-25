from __future__ import annotations

import pytest
import torch

from kandinsky_image_sr.attention import AttentionBackend, configure_attention

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


class RecordingTransformer:
    """Stand-in for a Diffusers model that records `set_attention_backend` calls."""

    def __init__(self, flash_installed: bool) -> None:
        self.flash_installed = flash_installed
        self.calls: list[str] = []

    def set_attention_backend(self, backend: str) -> None:
        self.calls.append(backend)
        if backend == "flash" and not self.flash_installed:
            raise RuntimeError("Flash Attention backend 'flash' is not usable because of missing package")


@pytest.mark.parametrize(
    ("requested", "flash_installed", "device", "expected", "calls"),
    [
        (AttentionBackend.AUTO, True, CUDA, AttentionBackend.FLASH, ["flash"]),
        (AttentionBackend.AUTO, False, CUDA, AttentionBackend.NATIVE, ["flash", "native"]),
        (AttentionBackend.AUTO, True, CPU, AttentionBackend.NATIVE, ["native"]),
        (AttentionBackend.FLASH, True, CUDA, AttentionBackend.FLASH, ["flash"]),
        (AttentionBackend.NATIVE, True, CUDA, AttentionBackend.NATIVE, ["native"]),
    ],
)
def test_configure_attention_selects_backend(
    requested: AttentionBackend,
    flash_installed: bool,
    device: torch.device,
    expected: AttentionBackend,
    calls: list[str],
) -> None:
    transformer = RecordingTransformer(flash_installed)

    assert configure_attention(transformer, requested, device) == expected
    assert transformer.calls == calls


def test_explicit_flash_without_package_raises() -> None:
    transformer = RecordingTransformer(flash_installed=False)

    with pytest.raises(RuntimeError, match="missing package"):
        configure_attention(transformer, AttentionBackend.FLASH, CUDA)


def test_explicit_flash_on_cpu_raises() -> None:
    transformer = RecordingTransformer(flash_installed=True)

    with pytest.raises(ValueError, match="requires a CUDA device"):
        configure_attention(transformer, AttentionBackend.FLASH, CPU)
    assert transformer.calls == []
