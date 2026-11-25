from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test.py"


@pytest.fixture(scope="module")
def benchmark() -> Iterator[ModuleType]:
    """Load scripts/test.py by path: the name `test` would resolve to the stdlib package."""
    spec = importlib.util.spec_from_file_location("benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.mark.parametrize(
    ("input_size", "upscale_factor", "expected"),
    [(1024, 2, 3), (1024, 4, 5), (2048, 2, 5), (2048, 4, 11), (4096, 2, 11), (128, 4, 1)],
)
def test_tile_grid_matches_pipeline_tiling(
    benchmark: ModuleType, input_size: int, upscale_factor: int, expected: int
) -> None:
    pipe = SimpleNamespace(config=SimpleNamespace(tile_size=1024, overlap=0.25))
    case = benchmark.BenchmarkCase(input_size=input_size, upscale_factor=upscale_factor)

    assert benchmark.tile_grid(pipe, case) == (expected, expected)


def test_square_image_center_crops_and_resizes(benchmark: ModuleType) -> None:
    image = Image.new("RGB", (300, 200))
    image.paste((255, 0, 0), (0, 0, 50, 200))

    square = benchmark.square_image(image, 64)

    assert square.size == (64, 64)
    assert square.getpixel((0, 32)) == (0, 0, 0)


def test_markdown_table_has_one_time_column_per_backend(benchmark: ModuleType) -> None:
    backends = [benchmark.AttentionBackend.NATIVE, benchmark.AttentionBackend.FLASH]
    case = benchmark.BenchmarkCase(input_size=1024, upscale_factor=2)
    results = [
        benchmark.BenchmarkResult(case=case, backend=backend, tile_grid=(3, 3), seconds=[3.0, 1.0, 2.0 + index])
        for index, backend in enumerate(backends)
    ]

    table = benchmark.markdown_table(results, backends).splitlines()

    assert table[0] == "| Input resolution | Upscale | Output resolution | Tile grid | SDPA, s | FlashAttention-2, s |"
    assert table[2] == "| 1K (1024 \u00d7 1024) | \u00d72 | 2K (2048 \u00d7 2048) | 3 \u00d7 3 | 2.00 | 3.00 |"
