from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import ACTION_SIZE, action_to_index


@dataclass(frozen=True)
class ResolvedAction:
    action_index: int
    actor_source: str
    metadata: dict


class ActionResolver:
    def resolve(self, action, observation: dict) -> ResolvedAction:
        if isinstance(action, (int, str)):
            return ResolvedAction(action_to_index(action), "physical", {})
        if not isinstance(action, dict):
            raise TypeError(f"unsupported action type: {type(action)!r}")

        kind = action.get("type", action.get("actor_source", "physical"))
        metadata = dict(action)
        if "action_index" in action:
            return ResolvedAction(action_to_index(action["action_index"]), str(kind), metadata)
        if "move" in action:
            return ResolvedAction(action_to_index(action["move"]), str(kind), metadata)
        if kind in {"ai", "katago_best"}:
            candidates = observation.get("top_candidates", [])
            if not candidates:
                raise ValueError("no KataGo candidate is available for ai action")
            return ResolvedAction(action_to_index(candidates[0]["move"]), str(kind), metadata)
        if kind in {"sample_policy", "policy_sample"}:
            policy = np.asarray(action["policy"], dtype=float)
            if policy.shape != (ACTION_SIZE,):
                raise ValueError(f"policy must have shape ({ACTION_SIZE},)")
            legal = np.asarray(observation["legal_mask"], dtype=bool)
            probs = np.where(legal, np.maximum(policy, 0), 0)
            total = probs.sum()
            if total <= 0:
                raise ValueError("policy has no legal probability mass")
            probs = probs / total
            rng = np.random.default_rng(action.get("seed"))
            return ResolvedAction(int(rng.choice(ACTION_SIZE, p=probs)), str(kind), metadata)
        raise ValueError(f"cannot resolve action: {action}")
