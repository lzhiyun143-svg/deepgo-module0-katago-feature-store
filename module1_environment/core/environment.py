from __future__ import annotations

from pathlib import Path

import numpy as np

from module0_katago_store.coords import coord_to_flat
from module0_katago_store.ids import position_id

from ..adapters.module0_adapter import Module0Adapter
from ..datasets.episode_manifest import EpisodeManifest, EpisodeSpec
from ..logging.transition_writer import TransitionWriter
from .action_resolver import ActionResolver
from .go_state import BLACK, WHITE, GoState
from .observation_builder import build_observation
from .transition import TransitionRecord


class MultiStepTaskEnv:
    def __init__(
        self,
        *,
        store_root: str | Path,
        episode_manifest: EpisodeManifest | None = None,
        transition_log_path: str | Path | None = None,
        max_steps: int | None = None,
    ):
        self.module0 = Module0Adapter(store_root)
        self.episodes = episode_manifest or EpisodeManifest.from_module0_store(store_root)
        self.max_steps = max_steps
        self.resolver = ActionResolver()
        self.writer = TransitionWriter(transition_log_path) if transition_log_path else None
        self.spec: EpisodeSpec | None = None
        self.state = GoState.empty()
        self.last_observation: dict | None = None

    def close(self) -> None:
        self.module0.close()
        if self.writer:
            self.writer.close()

    def reset(self, episode_spec: EpisodeSpec | None = None, seed: int | None = None) -> dict:
        del seed
        self.spec = episode_spec or self.episodes.get(0, max_steps=self.max_steps)
        self.state = GoState.empty(to_play=BLACK)
        for turn, (_, move) in enumerate(self.spec.moves[: self.spec.start_turn]):
            expected = BLACK if turn % 2 == 0 else WHITE
            self.state.to_play = expected
            result = self.state.apply(coord_to_flat(move))
            if not result.legal:
                raise ValueError(f"illegal replay move at turn {turn}: {move} ({result.reason})")
        self.last_observation = self._observation()
        return self.last_observation

    def offline_replay_action(self) -> int:
        if self.spec is None:
            raise RuntimeError("reset() must be called before offline_replay_action()")
        if self.state.turn_number >= len(self.spec.moves):
            return coord_to_flat("pass")
        _, move = self.spec.moves[self.state.turn_number]
        return coord_to_flat(move)

    def step(self, action, action_metadata: dict | None = None) -> tuple[dict, dict, bool, dict]:
        if self.spec is None or self.last_observation is None:
            raise RuntimeError("reset() must be called before step()")

        resolved = self.resolver.resolve(action, self.last_observation)
        metadata = {**resolved.metadata, **(action_metadata or {})}
        before_hash = self.state.state_hash()
        before_turn = self.state.turn_number
        before_position_id = self._position_id()

        result = self.state.apply(resolved.action_index)
        if not result.legal:
            info = {
                "legal": False,
                "reason": result.reason,
                "executed_action": resolved.action_index,
                "actor_source": resolved.actor_source,
                "position_id_before": before_position_id,
            }
            return self.last_observation, {"illegal_action": -1.0}, True, info

        done = self.state.done(self._episode_max_turn())
        after_position_id = self._position_id()
        after_hash = self.state.state_hash()
        next_obs = self._observation()
        reward_components = self._reward_components(self.last_observation, next_obs)
        record = TransitionRecord(
            episode_id=self.spec.episode_id,
            game_id=self.spec.game_id,
            turn_number_before=before_turn,
            position_id_before=before_position_id,
            position_id_after=after_position_id,
            action_index=resolved.action_index,
            action_gtp=result.action_gtp,
            actor_source=resolved.actor_source,
            legal=True,
            done=done,
            state_hash_before=before_hash,
            state_hash_after=after_hash,
            captured_count=len(result.captured),
            reward_components=reward_components,
            action_metadata=metadata,
        )
        if self.writer:
            self.writer.write(record)
        self.last_observation = next_obs
        info = record.to_dict()
        return next_obs, reward_components, done, info

    def _episode_max_turn(self) -> int | None:
        if self.spec is None:
            return None
        if self.spec.max_steps is None:
            return len(self.spec.moves)
        return min(len(self.spec.moves), self.spec.start_turn + self.spec.max_steps)

    def _position_id(self) -> str:
        if self.spec is None:
            raise RuntimeError("missing episode")
        return position_id(self.spec.game_id, self.state.turn_number)

    def _observation(self) -> dict:
        if self.spec is None:
            raise RuntimeError("missing episode")
        pid = self._position_id()
        features = None
        if self.module0.has_position(pid):
            features = self.module0.get_predecision_features(pid)
        return build_observation(
            state=self.state,
            episode_id=self.spec.episode_id,
            game_id=self.spec.game_id,
            position_id=pid,
            split=self.spec.split,
            module0_features=features,
            history_summary=self._history_summary(),
        )

    def _history_summary(self) -> np.ndarray:
        recent = self.state.move_history[-8:]
        pass_count = sum(1 for _, action in recent if action == 361)
        return np.asarray(
            [
                float(len(recent)),
                float(pass_count),
                float(self.state.consecutive_passes),
                float(self.state.turn_number),
            ],
            dtype=np.float32,
        )

    def _reward_components(self, before: dict, after: dict) -> dict[str, float]:
        before_winrate = _safe_float(before["katago_scalars"][0])
        after_winrate = _safe_float(after["katago_scalars"][0])
        if np.isnan(before_winrate) or np.isnan(after_winrate):
            return {}
        return {"delta_root_winrate": after_winrate - before_winrate}


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
