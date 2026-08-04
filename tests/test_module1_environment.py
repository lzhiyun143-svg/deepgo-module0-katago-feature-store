import json
from pathlib import Path

import numpy as np

from module0_katago_store.coords import coord_to_flat
from module0_katago_store.io import write_jsonl_atomic
from module0_katago_store.manifest import build_manifests
from module0_katago_store.normalize import normalize_responses
from module0_katago_store.profile import AnalysisProfile
from module0_katago_store.schema import REQUIRED_SCHEMA
from module1_environment import MultiStepTaskEnv
from module1_environment.core.go_state import GoState


def _make_store(tmp_path: Path) -> tuple[Path, list[dict]]:
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()
    (sgf_dir / "game1.sgf").write_text(
        "(;GM[1]FF[4]SZ[19]KM[7.5]PB[B]PW[W];B[pd];W[dp];B[qq])",
        encoding="utf-8",
    )
    store = tmp_path / "store" / "v1.0.0"
    profile = AnalysisProfile()
    profile.write(store)
    (store / "metadata").mkdir(parents=True, exist_ok=True)
    (store / "metadata" / "schema.json").write_text(json.dumps(REQUIRED_SCHEMA), encoding="utf-8")
    games, positions = build_manifests(sgf_dir, store)

    policy = np.full(362, 1 / 362, dtype=float)
    policy[coord_to_flat("Q16")] = 0.25
    policy = (policy / policy.sum()).tolist()
    responses = tmp_path / "responses.jsonl"
    write_jsonl_atomic(
        responses,
        [
            {
                "id": games[0]["game_id"],
                "turnNumber": 0,
                "policy": policy,
                "ownership": np.zeros((19, 19), dtype=float).reshape(-1).tolist(),
                "rootInfo": {"winrate": 0.55, "scoreLead": 1.2, "utility": 0.1, "visits": 25},
                "moveInfos": [{"move": "Q16", "prior": 0.2, "visits": 12, "winrate": 0.56, "scoreLead": 1.3}],
            },
            {
                "id": games[0]["game_id"],
                "turnNumber": 1,
                "policy": policy,
                "ownership": np.zeros((19, 19), dtype=float).reshape(-1).tolist(),
                "rootInfo": {"winrate": 0.50, "scoreLead": 0.3, "utility": 0.0, "visits": 25},
                "moveInfos": [{"move": "D4", "prior": 0.2, "visits": 12, "winrate": 0.51, "scoreLead": 0.4}],
            },
        ],
    )
    normalize_responses(store, responses, profile.__dict__)
    return store, positions


def test_go_state_legal_and_capture():
    state = GoState.empty()
    assert state.is_legal(coord_to_flat("Q16"))
    result = state.apply(coord_to_flat("Q16"))
    assert result.legal
    assert state.board.sum() == 1
    assert state.player_to_move_label == "W"
    assert state.board_tensor().shape == (17, 19, 19)


def test_module1_reset_step_and_module0_join(tmp_path: Path):
    store, positions = _make_store(tmp_path)
    log_path = tmp_path / "transitions.jsonl"
    env = MultiStepTaskEnv(store_root=store, transition_log_path=log_path, max_steps=2)
    obs = env.reset()
    assert obs["ids"]["position_id"] == positions[0]["position_id"]
    assert obs["board"].shape == (17, 19, 19)
    assert obs["legal_mask"].shape == (362,)
    assert obs["feature_mask"]["module0_hit"]
    assert obs["top_candidates"][0]["move"] == "Q16"

    next_obs, reward_components, done, info = env.step({"type": "katago_best"})
    assert info["legal"]
    assert info["actor_source"] == "katago_best"
    assert next_obs["ids"]["turn_number"] == 1
    assert isinstance(reward_components, dict)
    assert not done
    env.close()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["position_id_before"] == positions[0]["position_id"]
