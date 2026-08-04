from __future__ import annotations

import numpy as np


class RewardBuilder:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {
            "delta_root_winrate": 1.0,
            "delta_score_lead": 0.01,
            "illegal_action": -1.0,
            "switch_cost": -0.02,
        }

    def illegal(self) -> dict[str, float]:
        return {"illegal_action": self.weights.get("illegal_action", -1.0), "total": self.weights.get("illegal_action", -1.0)}

    def from_observations(self, before: dict, after: dict, *, actor_source: str, previous_actor_source: str | None) -> dict[str, float]:
        before_scalars = np.asarray(before.get("katago_scalars", []), dtype=float)
        after_scalars = np.asarray(after.get("katago_scalars", []), dtype=float)
        delta_winrate = _delta(before_scalars, after_scalars, 0)
        delta_score = _delta(before_scalars, after_scalars, 1)
        switch = 1.0 if previous_actor_source and previous_actor_source != actor_source else 0.0
        components = {
            "delta_root_winrate": delta_winrate,
            "delta_score_lead": delta_score,
            "switch_cost": switch * self.weights.get("switch_cost", -0.02),
        }
        total = 0.0
        if not np.isnan(delta_winrate):
            total += delta_winrate * self.weights.get("delta_root_winrate", 1.0)
        if not np.isnan(delta_score):
            total += delta_score * self.weights.get("delta_score_lead", 0.01)
        total += components["switch_cost"]
        components["total"] = float(total)
        return components


def _delta(before: np.ndarray, after: np.ndarray, idx: int) -> float:
    if len(before) <= idx or len(after) <= idx:
        return float("nan")
    if np.isnan(before[idx]) or np.isnan(after[idx]):
        return float("nan")
    return float(after[idx] - before[idx])
