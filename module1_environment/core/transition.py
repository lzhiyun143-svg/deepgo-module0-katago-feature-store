from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TransitionRecord:
    episode_id: str
    game_id: str
    turn_number_before: int
    position_id_before: str
    position_id_after: str
    action_index: int
    action_gtp: str
    actor_source: str
    legal: bool
    done: bool
    state_hash_before: str
    state_hash_after: str
    captured_count: int = 0
    reward_components: dict[str, float] = field(default_factory=dict)
    action_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
