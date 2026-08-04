from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class EnvironmentMode(str, Enum):
    OFFLINE_REPLAY = "offline_replay"
    SIMULATION = "simulation"
    LIVE_USER = "live_user"


class Delegation(str, Enum):
    HUMAN = "human"
    AI = "ai"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class EnvironmentConfig:
    mode: EnvironmentMode = EnvironmentMode.OFFLINE_REPLAY
    team_color: str = "B"
    auto_opponent: bool = False
    reward_weights: dict[str, float] = field(default_factory=dict)
    terminate_on_illegal: bool = True


class HumanModelProtocol(Protocol):
    def predict(self, observation: dict, player_profile: dict | None = None) -> dict:
        """Return a dict with at least one of policy[362], action_index, or move."""


class BeliefModelProtocol(Protocol):
    def update(self, observation: dict, transition: dict, previous_belief: dict | None = None) -> dict:
        """Return updated Module3 belief state."""


class LiveUserProtocol(Protocol):
    def get_action(self, observation: dict, metadata: dict | None = None) -> dict | int | str:
        """Return a physical Go action from a live UI or user event stream."""


class OpponentPolicyProtocol(Protocol):
    def predict(self, observation: dict, player_profile: dict | None = None) -> dict:
        """Return opponent action distribution or action."""


@dataclass
class StepResult:
    observation: dict
    reward_components: dict[str, float]
    done: bool
    info: dict[str, Any]
