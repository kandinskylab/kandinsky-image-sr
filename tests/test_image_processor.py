from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from kandinsky_image_sr.image_processor import image_to_tensor, tensor_to_pil


def test_pil_round_trip_preserves_rgb8_values() -> None:
    array = np.array([[[0, 127, 255], [4, 5, 6]]], dtype=np.uint8)
    tensor = image_to_tensor(Image.fromarray(array))
    assert tensor.shape == (1, 3, 1, 2)
    assert np.array_equal(np.asarray(tensor_to_pil(tensor)), array)


def test_tensor_input_accepts_uint8_chw() -> None:
    tensor = image_to_tensor(torch.full((3, 2, 4), 255, dtype=torch.uint8))
    assert tensor.shape == (1, 3, 2, 4)
    assert tensor.min() == tensor.max() == 1


def test_rejects_batched_inputs() -> None:
    with pytest.raises(ValueError, match="expected one RGB image"):
        image_to_tensor(torch.zeros(2, 3, 8, 8))
