from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kandinsky_image_sr.pipeline import KandinskyImageSRPipeline


class Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace()


def make_pipeline() -> KandinskyImageSRPipeline:
    vae = Stub()
    vae.config.scaling_factor = 1.0
    return KandinskyImageSRPipeline(transformer=Stub(), vae=vae)


def test_default_timesteps() -> None:
    pipe = make_pipeline()
    with pytest.raises(ValueError, match="no default timestep configured for x1"):
        pipe.default_timestep(1)
    assert pipe.default_timestep(2) == 0.15
    assert pipe.default_timestep(4) == 0.244


def test_raw_timestep_is_public_and_validated() -> None:
    pipe = make_pipeline()
    assert pipe._resolve_timestep(4, 0.31, None) == 0.31
    with pytest.raises(ValueError, match="must lie"):
        pipe._resolve_timestep(4, 1.0, None)


def test_baked_model_restricts_raw_timestep_to_saved_table() -> None:
    pipe = make_pipeline()
    pipe.transformer.config.baked_timesteps = [0.15, 0.244]
    assert pipe._resolve_timestep(4, 0.244, None) == 0.244
    with pytest.raises(ValueError, match="not present"):
        pipe._resolve_timestep(4, 0.2, None)


def test_creativity_presets_and_raw_t_are_mutually_exclusive() -> None:
    pipe = make_pipeline()
    assert pipe._resolve_timestep(2, None, 0) == 0.07
    assert pipe._resolve_timestep(4, None, 8) == 0.38
    with pytest.raises(ValueError, match="mutually exclusive"):
        pipe._resolve_timestep(4, 0.2, 3)
    with pytest.raises(ValueError, match="only for x2 and x4"):
        pipe._resolve_timestep(1, None, 4)


def test_model_dtype_follows_visual_path_for_mixed_precision_transformer() -> None:
    transformer = Stub()
    transformer.visual_embeddings = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    vae = Stub()
    vae.config.scaling_factor = 1.0
    pipe = KandinskyImageSRPipeline(transformer=transformer, vae=vae)
    assert pipe._model_dtype() == torch.bfloat16
