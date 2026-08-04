from __future__ import annotations

import numpy as np

from module0_katago_store.coords import BOARD_SIZE, PASS_INDEX, coord_to_flat, flat_to_coord

ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1


def action_to_index(action: int | str) -> int:
    if isinstance(action, str):
        return coord_to_flat(action)
    index = int(action)
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"action index out of range: {index}")
    return index


def index_to_action(index: int) -> str:
    return flat_to_coord(index)


def flatten_legal_mask(mask_2d: np.ndarray) -> np.ndarray:
    flat = np.zeros(ACTION_SIZE, dtype=bool)
    flat[: BOARD_SIZE * BOARD_SIZE] = np.asarray(mask_2d, dtype=bool).reshape(-1)
    flat[PASS_INDEX] = True
    return flat
