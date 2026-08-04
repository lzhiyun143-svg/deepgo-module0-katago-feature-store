from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module1_environment.core.action_space import ACTION_SIZE
from module1_environment.core.types import HumanModelProtocol


@dataclass
class HumanPrediction:
    policy: np.ndarray | None = None
    action_index: int | None = None
    value: float | None = None
    uncertainty: float | None = None
    raw: dict | None = None


class Module2Adapter:
    """Thin adapter for a future personalized human decision model."""

    def __init__(self, model: HumanModelProtocol | None = None):
        self.model = model

    def predict(self, observation: dict, player_profile: dict | None = None) -> HumanPrediction:
        if self.model is None:
            legal = np.asarray(observation["legal_mask"], dtype=bool)
            policy = legal.astype(np.float32)
            policy = policy / max(float(policy.sum()), 1.0)
            return HumanPrediction(policy=policy, raw={"source": "uniform_legal_stub"})
        raw = self.model.predict(observation, player_profile=player_profile)
        policy = raw.get("policy")
        if policy is not None:
            policy = np.asarray(policy, dtype=np.float32)
            if policy.shape != (ACTION_SIZE,):
                raise ValueError(f"Module2 policy must have shape ({ACTION_SIZE},), got {policy.shape}")
        return HumanPrediction(
            policy=policy,
            action_index=raw.get("action_index"),
            value=raw.get("value"),
            uncertainty=raw.get("uncertainty"),
            raw=raw,
        )
