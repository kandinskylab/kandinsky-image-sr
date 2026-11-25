from __future__ import annotations

from diffusers import AutoencoderKL, Kandinsky5Transformer3DModel

from kandinsky_image_sr import KandinskyImageSRPipeline


def tiny_transformer() -> Kandinsky5Transformer3DModel:
    return Kandinsky5Transformer3DModel(
        in_visual_dim=2,
        in_text_dim=8,
        in_text_dim2=4,
        time_dim=8,
        out_visual_dim=2,
        patch_size=(1, 2, 2),
        model_dim=12,
        ff_dim=24,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(2, 2, 2),
        visual_cond=False,
        attention_type="regular",
    )


def tiny_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=8,
        scaling_factor=1.0,
    )


def test_save_pretrained_round_trip_uses_diffusers_components(tmp_path) -> None:
    pipe = KandinskyImageSRPipeline(tiny_transformer(), tiny_vae(), tile_size=8)
    pipe.save_pretrained(tmp_path, safe_serialization=True)

    assert (tmp_path / "model_index.json").is_file()
    assert list((tmp_path / "transformer").glob("*.safetensors"))
    assert list((tmp_path / "vae").glob("*.safetensors"))

    loaded = KandinskyImageSRPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded.transformer, Kandinsky5Transformer3DModel)
    assert isinstance(loaded.vae, AutoencoderKL)
    assert loaded.upscale_factors == (2, 4)
    assert loaded.default_timestep(4) == 0.244
    assert loaded.config.tile_size == 8
