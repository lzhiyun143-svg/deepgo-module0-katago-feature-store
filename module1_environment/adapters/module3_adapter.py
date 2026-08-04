from __future__ import annotations

from module1_environment.core.types import BeliefModelProtocol


class Module3Adapter:
    """Adapter for the future latent human-state / belief update module."""

    def __init__(self, model: BeliefModelProtocol | None = None):
        self.model = model

    def initial_belief(self) -> dict:
        return {
            "coherence": 0.0,
            "understanding": 0.0,
            "readiness": 0.0,
            "agency_alignment": 0.0,
            "trust_alignment": 0.0,
            "source": "empty_stub",
        }

    def update(self, observation: dict, transition: dict, previous_belief: dict | None = None) -> dict:
        if self.model is None:
            belief = dict(previous_belief or self.initial_belief())
            belief["last_actor_source"] = transition.get("actor_source")
            belief["last_legal"] = transition.get("legal")
            return belief
        return self.model.update(observation, transition, previous_belief=previous_belief)
