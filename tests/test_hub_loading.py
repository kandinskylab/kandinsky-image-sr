from __future__ import annotations

from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, _get_custom_components_and_folders

import kandinsky_image_sr

RELEASE_MODEL_INDEX = {
    "_class_name": "KandinskyImageSRPipeline",
    "transformer": ["kandinsky_image_sr", "KandinskyImageSRBakedTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}
RELEASE_FILES = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def test_hub_download_accepts_package_transformer_component() -> None:
    """Diffusers validates component libraries before downloading a Hub repo id."""
    custom_components, folder_names = _get_custom_components_and_folders(
        "kandinskylab/Kandinsky-ImageSR-2.0", RELEASE_MODEL_INDEX, RELEASE_FILES
    )

    assert custom_components == {}
    assert sorted(folder_names) == ["transformer", "vae"]
    assert LOADABLE_CLASSES[kandinsky_image_sr.__name__] == {"ModelMixin": ["save_pretrained", "from_pretrained"]}
