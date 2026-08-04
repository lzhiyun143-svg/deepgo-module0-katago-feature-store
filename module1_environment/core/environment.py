from __future__ import annotations

from pathlib import Path

import numpy as np

from module0_katago_store.coords import coord_to_flat
from module0_katago_store.ids import position_id

from ..adapters.module2_adapter import Module2Adapter
from ..adapters.module3_adapter import Module3Adapter
from ..adapters.module4_adapter import normalize_delegation
from ..adapters.module0_adapter import Module0Adapter
from ..datasets.episode_manifest import EpisodeManifest, EpisodeSpec
from ..logging.transition_writer import TransitionWriter
from .action_resolver import ActionResolver
from .go_state import BLACK, WHITE, GoState
from .observation_builder import build_observation
from .opponent_driver import OpponentDriver
from .reward import RewardBuilder
from .transition import TransitionRecord
from .types import Delegation, EnvironmentConfig, EnvironmentMode


class MultiStepTaskEnv:
    def __init__(
        self,
        *,
        store_root: str | Path,
        episode_manifest: EpisodeManifest | None = None,
        transition_log_path: str | Path | None = None,
        max_steps: int | None = None,
        config: EnvironmentConfig | None = None,
        human_model=None,
        belief_model=None,
        opponent_model=None,
        player_profile: dict | None = None,
    ):
        self.config = config or EnvironmentConfig()
        self.module0 = Module0Adapter(store_root)
        self.module2 = Module2Adapter(human_model)
        self.module3 = Module3Adapter(belief_model)
        self.episodes = episode_manifest or EpisodeManifest.from_module0_store(store_root)
        self.max_steps = max_steps
        self.player_profile = player_profile
        self.opponent_driver = OpponentDriver(opponent_model)
        self.reward_builder = RewardBuilder(self.config.reward_weights)
        self.resolver = ActionResolver()
        self.writer = TransitionWriter(transition_log_path) if transition_log_path else None
        self.spec: EpisodeSpec | None = None
        self.state = GoState.empty()
        self.last_observation: dict | None = None
        self.belief: dict | None = None
        self.previous_actor_source: str | None = None

    def close(self) -> None:
        self.module0.close()
        if self.writer:
            self.writer.close()

    def reset(self, episode_spec: EpisodeSpec | None = None, seed: int | None = None) -> dict:
        del seed
        self.spec = episode_spec or self.episodes.get(0, max_steps=self.max_steps)
        self.state = GoState.empty(to_play=BLACK)
        self.belief = self.module3.initial_belief()
        self.previous_actor_source = None
        for turn, (_, move) in enumerate(self.spec.moves[: self.spec.start_turn]):
            expected = BLACK if turn % 2 == 0 else WHITE
            self.state.to_play = expected
            result = self.state.apply(coord_to_flat(move))
            if not result.legal:
                raise ValueError(f"illegal replay move at turn {turn}: {move} ({result.reason})")
        self.last_observation = self._observation()
        self.last_observation["belief"] = self.belief
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

        meta_decision = self._normalize_meta_decision(action, action_metadata)
        resolved = self._resolve_team_action(meta_decision)
        metadata = {**resolved.metadata, **(action_metadata or {})}
        before_hash = self.state.state_hash()
        before_turn = self.state.turn_number
        before_position_id = self._position_id()

        result = self.state.apply(resolved.action_index)
        if not result.legal:
            reward = self.reward_builder.illegal()
            info = {
                "legal": False,
                "reason": result.reason,
                "executed_action": resolved.action_index,
                "actor_source": resolved.actor_source,
                "position_id_before": before_position_id,
                "reward_components": reward,
                "meta_decision": meta_decision,
            }
            return self.last_observation, reward, self.config.terminate_on_illegal, info

        intermediate_position_id = self._position_id()
        opponent_info = None
        done = self.state.done(self._episode_max_turn())
        if self.config.auto_opponent and not done:
            opponent_obs = self._observation()
            opponent = self.opponent_driver.choose_action(opponent_obs, player_profile=self.player_profile)
            opp_result = self.state.apply(opponent.action_index)
            opponent_info = {
                "action_index": opponent.action_index,
                "action_gtp": opp_result.action_gtp,
                "actor_source": opponent.actor_source,
                "legal": opp_result.legal,
                "captured_count": len(opp_result.captured) if opp_result.legal else 0,
                "reason": opp_result.reason,
            }
            if not opp_result.legal:
                done = self.config.terminate_on_illegal

        done = done or self.state.done(self._episode_max_turn())
        after_position_id = self._position_id()
        after_hash = self.state.state_hash()
        next_obs = self._observation()
        reward_components = self.reward_builder.from_observations(
            self.last_observation,
            next_obs,
            actor_source=resolved.actor_source,
            previous_actor_source=self.previous_actor_source,
        )
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
            meta_decision=meta_decision,
            physical_action={
                "action_index": resolved.action_index,
                "action_gtp": result.action_gtp,
                "actor_source": resolved.actor_source,
                "captured_count": len(result.captured),
                "intermediate_position_id": intermediate_position_id,
            },
            opponent_action=opponent_info,
            interaction_metadata=metadata,
        )
        self.belief = self.module3.update(next_obs, record.to_dict(), previous_belief=self.belief)
        record.module3_belief = self.belief
        if self.writer:
            self.writer.write(record)
        next_obs["belief"] = self.belief
        self.last_observation = next_obs
        self.previous_actor_source = resolved.actor_source
        info = record.to_dict()
        return next_obs, reward_components, done, info

    def _normalize_meta_decision(self, action, action_metadata: dict | None) -> dict:
        if isinstance(action, dict) and any(k in action for k in ("delegation", "type", "actor_source")):
            meta = dict(action)
        else:
            meta = {"delegation": Delegation.PHYSICAL.value, "action_index": action}
        if action_metadata:
            meta.update(action_metadata)
        return meta

    def _resolve_team_action(self, meta_decision: dict):
        decision = normalize_delegation(meta_decision)
        delegation = decision.delegation
        if delegation == Delegation.AI.value:
            return self.resolver.resolve({"actor_source": "ai", "type": "katago_best"}, self.last_observation)
        if delegation == "katago_best":
            return self.resolver.resolve({"actor_source": "katago_best", "type": "katago_best"}, self.last_observation)
        if delegation in {Delegation.HUMAN.value, "human_model"}:
            if self.config.mode == EnvironmentMode.OFFLINE_REPLAY:
                return self.resolver.resolve(
                    {"actor_source": "offline_replay", "action_index": self.offline_replay_action()},
                    self.last_observation,
                )
            prediction = self.module2.predict(self.last_observation, player_profile=self.player_profile)
            if prediction.action_index is not None:
                return self.resolver.resolve(
                    {"actor_source": "human_model", "action_index": prediction.action_index, "prediction": prediction.raw},
                    self.last_observation,
                )
            if prediction.policy is not None:
                return self.resolver.resolve(
                    {"actor_source": "human_model", "type": "policy_sample", "policy": prediction.policy},
                    self.last_observation,
                )
            raise ValueError("Module2 did not return action_index or policy")
        return self.resolver.resolve(meta_decision, self.last_observation)

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

