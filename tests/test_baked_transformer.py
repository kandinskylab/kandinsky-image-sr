from __future__ import annotations

import pytest
import torch
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import AttentionBackendName

from kandinsky_image_sr import KandinskyImageSRBakedTransformer3DModel


def tiny_baked_transformer() -> KandinskyImageSRBakedTransformer3DModel:
    return KandinskyImageSRBakedTransformer3DModel(
        in_visual_dim=2,
        out_visual_dim=2,
        time_dim=8,
        patch_size=(1, 2, 2),
        model_dim=12,
        ff_dim=24,
        num_visual_blocks=1,
        axes_dims=(2, 2, 2),
        visual_cond=False,
        attention_type="regular",
        baked_timesteps=(0.1, 0.2),
        baked_latent_size=4,
    )


def model_inputs(timestep: float) -> dict[str, object]:
    return {
        "hidden_states": torch.randn(1, 1, 4, 4, 2),
        "encoder_hidden_states": None,
        "timestep": torch.tensor([timestep * 1000.0]),
        "pooled_projections": None,
        "visual_rope_pos": (torch.arange(1), torch.arange(2), torch.arange(2)),
        "text_rope_pos": None,
        "return_dict": False,
    }


def test_baked_transformer_round_trip_and_forward(tmp_path) -> None:
    model = tiny_baked_transformer().eval()
    model.time_embed_table.normal_()
    model.visual_transformer_blocks[0].xattn_bias_table.normal_()
    inputs = model_inputs(0.2)
    expected = model(**inputs)

    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = KandinskyImageSRBakedTransformer3DModel.from_pretrained(tmp_path).eval()
    actual = loaded(**inputs)

    torch.testing.assert_close(actual, expected)
    keys = set(loaded.state_dict())
    assert "time_embed_table" in keys
    assert "visual_transformer_blocks.0.xattn_bias_table" in keys
    assert not any("cross_attention" in key or key.startswith("text_") for key in keys)

    mixed = KandinskyImageSRBakedTransformer3DModel.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert mixed.time_embed_table.dtype == torch.float32
    assert mixed.visual_transformer_blocks[0].xattn_bias_table.dtype == torch.float32


def test_baked_transformer_rejects_unavailable_timestep() -> None:
    model = tiny_baked_transformer().eval()
    with pytest.raises(ValueError, match="not baked"):
        model(**model_inputs(0.15))


def test_baked_transformer_switches_attention_backend() -> None:
    torch.manual_seed(0)
    model = tiny_baked_transformer().eval()
    model.time_embed_table.normal_()
    inputs = model_inputs(0.1)
    with torch.inference_mode():
        expected = model(**inputs)[0]
        model.set_attention_backend("_native_math")
        actual = model(**inputs)[0]

    backends = {
        module.processor._attention_backend for module in model.modules() if isinstance(module, AttentionModuleMixin)
    }
    assert backends == {AttentionBackendName._NATIVE_MATH}
    torch.testing.assert_close(actual, expected)
