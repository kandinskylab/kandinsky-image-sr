#!/usr/bin/env python3
"""Super-resolve one image with Kandinsky ImageSR 2.0.

FlashAttention-2 is used when the `flash` extra is installed and the device is CUDA;
otherwise the pipeline runs on PyTorch SDPA. Force either with `--attention`.

Example:
    python scripts/run_inference.py --input input.png --output output.png --upscale-factor 4
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, Field

from kandinsky_image_sr import KandinskyImageSRPipeline
from kandinsky_image_sr.attention import AttentionBackend, configure_attention

DEFAULT_MODEL = "kandinskylab/Kandinsky-ImageSR-2.0"


class InferenceConfig(BaseModel):
    """Settings for a single-image super-resolution run."""

    input_path: Path
    output_path: Path
    model: str = DEFAULT_MODEL
    upscale_factor: Literal[2, 4] = 4
    creativity: int | None = Field(default=None, ge=0, le=8)
    one_step_t: float | None = None
    attention: AttentionBackend = AttentionBackend.AUTO
    device: str = "cuda"
    seed: int = 0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pipeline(config: InferenceConfig) -> KandinskyImageSRPipeline:
    """Load the pipeline in bfloat16 and switch it to the requested attention backend."""
    device = torch.device(config.device)
    pipe = KandinskyImageSRPipeline.from_pretrained(config.model, dtype=torch.bfloat16).to(device)
    backend = configure_attention(pipe.transformer, config.attention, device)
    logger.info("Attention backend: {} (requested: {})", backend.value, config.attention.value)
    return pipe


def run(config: InferenceConfig) -> None:
    """Super-resolve `config.input_path` and save the result to `config.output_path`."""
    seed_everything(config.seed)
    pipe = load_pipeline(config)
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            str(config.input_path),
            upscale_factor=config.upscale_factor,
            one_step_t=config.one_step_t,
            creativity=config.creativity,
        )
    image = result.images[0]
    logger.info("Upscaled x{} to {}x{} in {:.2f} s", config.upscale_factor, *image.size, time.perf_counter() - start)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(config.output_path)
    logger.info("Saved {}", config.output_path)


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="HF repo id or local bundle path.")
@click.option("--upscale-factor", type=click.Choice(["2", "4"]), default="4", show_default=True)
@click.option("--creativity", type=click.IntRange(0, 8), default=None, help="Calibrated level 0..8.")
@click.option("--one-step-t", type=float, default=None, help="Raw timestep; mutually exclusive with --creativity.")
@click.option(
    "--attention",
    type=click.Choice([backend.value for backend in AttentionBackend]),
    default=AttentionBackend.AUTO.value,
    show_default=True,
    help="auto: FlashAttention-2 when available, PyTorch SDPA otherwise.",
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(**options: object) -> None:
    """Super-resolve one image with Kandinsky ImageSR 2.0."""
    options["upscale_factor"] = int(str(options["upscale_factor"]))
    run(InferenceConfig.model_validate(options))


if __name__ == "__main__":
    main()
