#!/usr/bin/env python3
"""Measure Kandinsky ImageSR 2.0 inference time per image, as in the README table.

The test image is center-cropped to a square and resized to each input resolution. Every
case runs warm-up calls first, then timed calls bracketed by `torch.cuda.synchronize`;
the reported time is the median of the timed calls for one full `pipe(...)` call.

Example:
    CUDA_VISIBLE_DEVICES=1 python scripts/test.py --image photo.jpg
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from kandinsky_image_sr import KandinskyImageSRPipeline
from kandinsky_image_sr.attention import AttentionBackend, configure_attention
from kandinsky_image_sr.tiling import tile_positions

DEFAULT_MODEL = "kandinskylab/Kandinsky-ImageSR-2.0"
RESOLUTION_LABELS = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K"}
TIMES = "\u00d7"  # the multiplication sign used by the README table


class BenchmarkCase(BaseModel):
    """One row of the timing table: a square input side and an upscale factor."""

    input_size: int
    upscale_factor: int


DEFAULT_CASES = [
    BenchmarkCase(input_size=1024, upscale_factor=2),
    BenchmarkCase(input_size=1024, upscale_factor=4),
    BenchmarkCase(input_size=2048, upscale_factor=2),
    BenchmarkCase(input_size=2048, upscale_factor=4),
    BenchmarkCase(input_size=4096, upscale_factor=2),
]


class BenchmarkConfig(BaseModel):
    """Settings for a timing run."""

    image_path: Path
    model: str = DEFAULT_MODEL
    cases: list[BenchmarkCase] = DEFAULT_CASES
    backends: list[AttentionBackend] = [AttentionBackend.NATIVE, AttentionBackend.FLASH]
    warmup: int = 2
    runs: int = 5
    device: str = "cuda"
    seed: int = 0


class BenchmarkResult(BaseModel):
    """Timing of one case on one attention backend."""

    case: BenchmarkCase
    backend: AttentionBackend
    tile_grid: tuple[int, int]
    seconds: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def square_image(image: Image.Image, size: int) -> Image.Image:
    """Center-crop `image` to a square and resize it to `size` x `size`."""
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def tile_grid(pipe: KandinskyImageSRPipeline, case: BenchmarkCase) -> tuple[int, int]:
    """Rows and columns of output tiles the pipeline processes for a square input."""
    lq_tile = int(pipe.config.tile_size) // case.upscale_factor
    stride = max(1, int(lq_tile * (1.0 - float(pipe.config.overlap))))
    count = len(tile_positions(max(case.input_size, lq_tile), lq_tile, stride))
    return count, count


def time_call(pipe: KandinskyImageSRPipeline, image: Image.Image, upscale_factor: int, device: torch.device) -> float:
    """Wall-clock seconds of one synchronized pipeline call."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        pipe(image, upscale_factor=upscale_factor)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def benchmark_case(
    pipe: KandinskyImageSRPipeline, image: Image.Image, case: BenchmarkCase, config: BenchmarkConfig
) -> list[float]:
    """Run warm-up calls, then return the timings of the measured calls."""
    device = torch.device(config.device)
    lq = square_image(image, case.input_size)
    for _ in range(config.warmup):
        time_call(pipe, lq, case.upscale_factor, device)
    return [time_call(pipe, lq, case.upscale_factor, device) for _ in range(config.runs)]


def benchmark_backend(
    pipe: KandinskyImageSRPipeline, image: Image.Image, backend: AttentionBackend, config: BenchmarkConfig
) -> list[BenchmarkResult]:
    """Time every case on one attention backend."""
    configure_attention(pipe.transformer, backend, torch.device(config.device))
    results = []
    for case in config.cases:
        seconds = benchmark_case(pipe, image, case, config)
        result = BenchmarkResult(case=case, backend=backend, tile_grid=tile_grid(pipe, case), seconds=seconds)
        logger.info(
            "{} {}px x{}: median {:.2f} s, runs {}",
            backend.value,
            case.input_size,
            case.upscale_factor,
            result.median,
            [round(value, 2) for value in seconds],
        )
        results.append(result)
    return results


def resolution(side: int) -> str:
    """README-style resolution label, e.g. `1K (1024 x 1024)` with a multiplication sign."""
    return f"{RESOLUTION_LABELS.get(side, f'{side}px')} ({side} {TIMES} {side})"


def markdown_table(results: list[BenchmarkResult], backends: list[AttentionBackend]) -> str:
    """Timing table with one time column per attention backend."""
    names = {AttentionBackend.NATIVE: "SDPA", AttentionBackend.FLASH: "FlashAttention-2"}
    header = "| Input resolution | Upscale | Output resolution | Tile grid | "
    header += " | ".join(f"{names[backend]}, s" for backend in backends) + " |"
    separator = "|---|:---:|---|:---:|" + "---:|" * len(backends)
    by_key = {(r.case.input_size, r.case.upscale_factor, r.backend): r for r in results}
    rows = []
    for case in dict.fromkeys((r.case.input_size, r.case.upscale_factor) for r in results):
        first = by_key[(*case, backends[0])]
        times = " | ".join(f"{by_key[(*case, backend)].median:.2f}" for backend in backends)
        rows.append(
            f"| {resolution(case[0])} | {TIMES}{case[1]} | {resolution(case[0] * case[1])} "
            f"| {first.tile_grid[0]} {TIMES} {first.tile_grid[1]} | {times} |"
        )
    return "\n".join([header, separator, *rows])


def run(config: BenchmarkConfig) -> None:
    """Load the pipeline once, time every backend and case, and print the table."""
    seed_everything(config.seed)
    device = torch.device(config.device)
    pipe = KandinskyImageSRPipeline.from_pretrained(config.model, dtype=torch.bfloat16).to(device)
    logger.info("GPU: {}, torch {}", torch.cuda.get_device_name(device), torch.__version__)
    image = Image.open(config.image_path).convert("RGB")
    results = [result for backend in config.backends for result in benchmark_backend(pipe, image, backend, config)]
    click.echo(markdown_table(results, config.backends))


@click.command()
@click.option("--image", "image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="HF repo id or local bundle path.")
@click.option(
    "--backend",
    "backends",
    type=click.Choice([AttentionBackend.NATIVE.value, AttentionBackend.FLASH.value]),
    multiple=True,
    default=[AttentionBackend.NATIVE.value, AttentionBackend.FLASH.value],
    show_default=True,
    help="Attention backends to time; repeat the option for several.",
)
@click.option("--warmup", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--runs", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(**options: object) -> None:
    """Measure inference time per image for the README timing table."""
    run(BenchmarkConfig.model_validate(options))


if __name__ == "__main__":
    main()
