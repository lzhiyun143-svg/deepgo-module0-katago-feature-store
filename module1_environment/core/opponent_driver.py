from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import ACTION_SIZE
from .action_resolver import ActionResolver, ResolvedAction


@dataclass
class OpponentDriver:
    policy_model: object | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.resolver = ActionResolver()
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, observation: dict, player_profile: dict | None = None) -> ResolvedAction:
        if self.policy_model is not None:
            pred = self.policy_model.predict(observation, player_profile=player_profile)
            if "action_index" in pred:
                return self.resolver.resolve(
                    {"type": "opponent_model", "action_index": pred["action_index"], "prediction": pred},
                    observation,
                )
            if "move" in pred:
                return self.resolver.resolve(
                    {"type": "opponent_model", "move": pred["move"], "prediction": pred},
                    observation,
                )
            if "policy" in pred:
                return self.resolver.resolve(
                    {
                        "type": "policy_sample",
                        "policy": pred["policy"],
                        "seed": int(self.rng.integers(0, 2**31 - 1)),
                        "actor_source": "opponent_model",
                        "prediction": pred,
                    },
                    observation,
                )

        candidates = observation.get("top_candidates", [])
        if candidates:
            return self.resolver.resolve({"type": "opponent_katago_best"}, observation)

        legal = np.asarray(observation["legal_mask"], dtype=bool)
        policy = legal.astype(np.float32)
        policy = policy / max(float(policy.sum()), 1.0)
        return self.resolver.resolve(
            {
                "type": "policy_sample",
                "policy": policy,
                "seed": int(self.rng.integers(0, 2**31 - 1)),
                "actor_source": "opponent_uniform",
            },
            observation,
        )
