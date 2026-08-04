from __future__ import annotations

import numpy as np

from .action_space import flatten_legal_mask
from .go_state import GoState


def build_observation(
    *,
    state: GoState,
    episode_id: str,
    game_id: str,
    position_id: str,
    split: str,
    module0_features: dict | None,
    history_summary: np.ndarray | None = None,
) -> dict:
    legal_mask_2d = state.legal_mask()
    kg = module0_features or {}
    scalars = np.asarray(
        [
            _num(kg.get("root_winrate")),
            _num(kg.get("root_score_lead")),
            _num(kg.get("root_utility")),
            _num(kg.get("policy_entropy")),
            _num(kg.get("policy_gap_top1_top2")),
        ],
        dtype=np.float32,
    )
    task_scalars = np.asarray(
        [
            float(state.turn_number),
            float(state.turn_number / 361.0),
            1.0 if state.to_play == 1 else -1.0,
        ],
        dtype=np.float32,
    )
    has_ownership = kg.get("ownership_map") is not None
    ownership = kg.get("ownership_map")
    if not has_ownership:
        ownership = np.full((19, 19), np.nan, dtype=np.float32)

    return {
        "ids": {
            "episode_id": episode_id,
            "game_id": game_id,
            "position_id": position_id,
            "turn_number": state.turn_number,
            "split": split,
        },
        "board": state.board_tensor(),
        "legal_mask": flatten_legal_mask(legal_mask_2d),
        "task_scalars": task_scalars,
        "katago_spatial": {"ownership": np.asarray(ownership, dtype=np.float32)},
        "katago_scalars": scalars,
        "top_candidates": kg.get("top_candidates", []),
        "history_summary": history_summary if history_summary is not None else np.zeros(4, dtype=np.float32),
        "feature_mask": {
            "module0_hit": bool(module0_features),
            "has_ownership": has_ownership,
        },
    }


def _num(value) -> float:
    if value is None:
        return float("nan")
    return float(value)
