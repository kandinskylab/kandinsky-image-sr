from __future__ import annotations

import torch

from kandinsky_image_sr.tiling import boundary_aware_window, pad_to_min_tile, tile_positions


def test_tile_positions_snap_last_tile_to_edge() -> None:
    assert tile_positions(100, 32, 24) == [0, 24, 48, 68]


def test_pad_to_min_tile_replicates_and_records_original_shape() -> None:
    image = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3)
    padded, height, width = pad_to_min_tile(image, 4)
    assert (height, width) == (2, 3)
    assert padded.shape == (1, 3, 4, 4)
    assert torch.equal(padded[:, :, -1, -1], image[:, :, -1, -1])


def test_boundary_window_keeps_image_edges_nonzero() -> None:
    window = boundary_aware_window(16, 16, top=True, bottom=False, left=True, right=False, device=torch.device("cpu"))
    assert window[0, 0] == 1
    assert window[-1, -1] > 0
