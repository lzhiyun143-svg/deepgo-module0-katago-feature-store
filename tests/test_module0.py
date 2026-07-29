import json
from pathlib import Path

import numpy as np

from module0_katago_store import FeatureStore
from module0_katago_store.analysis import run_katago_analysis_streaming
from module0_katago_store.coords import coord_to_flat, coord_to_rc, flat_to_coord, rc_to_coord
from module0_katago_store.io import write_jsonl_atomic
from module0_katago_store.manifest import build_manifests
from module0_katago_store.normalize import normalize_responses
from module0_katago_store.profile import AnalysisProfile
from module0_katago_store.qa import run_qa
from module0_katago_store.schema import REQUIRED_SCHEMA


def test_coordinates():
    assert coord_to_rc("A19") == (0, 0)
    assert coord_to_rc("T1") == (18, 18)
    assert rc_to_coord(0, 0) == "A19"
    assert flat_to_coord(coord_to_flat("Q16")) == "Q16"
    assert coord_to_flat("pass") == 361


def test_end_to_end_store(tmp_path: Path):
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()
    (sgf_dir / "game1.sgf").write_text("(;GM[1]FF[4]SZ[19]KM[7.5]PB[B]PW[W];B[pd];W[dp])", encoding="utf-8")
    store = tmp_path / "store" / "v1.0.0"
    profile = AnalysisProfile()
    profile.write(store)
    (store / "metadata").mkdir(parents=True, exist_ok=True)
    import json

    (store / "metadata" / "schema.json").write_text(json.dumps(REQUIRED_SCHEMA), encoding="utf-8")
    games, positions = build_manifests(sgf_dir, store)

    policy = np.full(362, 1 / 362, dtype=float).tolist()
    responses = tmp_path / "responses.jsonl"
    write_jsonl_atomic(
        responses,
        [
            {
                "id": games[0]["game_id"],
                "turnNumber": 0,
                "policy": policy,
                "ownership": np.zeros((19, 19), dtype=float).reshape(-1).tolist(),
                "rootInfo": {"winrate": 0.55, "scoreLead": 1.2, "utility": 0.1, "visits": 100},
                "moveInfos": [{"move": "Q16", "prior": 0.1, "visits": 50, "winrate": 0.56, "scoreLead": 1.3}],
            }
        ],
    )
    report = normalize_responses(store, responses, profile.__dict__)
    assert report["positions_written"] == 1
    qa = run_qa(store)
    assert qa["passed"]

    fs = FeatureStore(store)
    pid = positions[0]["position_id"]
    feat = fs.get(pid, fields=["policy_map", "ownership_map", "root_winrate", "human_move_rank_policy"])
    fs.close()
    assert feat["policy_map"].shape == (19, 19)
    assert feat["ownership_map"].shape == (19, 19)
    assert feat["root_winrate"] == 0.55


def test_streaming_analysis_uses_one_engine_process_and_resumes(tmp_path: Path):
    engine = tmp_path / "fake_katago.py"
    engine.write_text(
        """#!/usr/bin/env python3
import json
import sys

print("engine started", file=sys.stderr, flush=True)
for line in sys.stdin:
    query = json.loads(line)
    for turn in query.get("analyzeTurns", [len(query.get("moves", []))]):
        print(json.dumps({"id": query["id"], "turnNumber": turn, "isDuringSearch": False}), flush=True)
""",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            [
                json.dumps({"id": "game-a", "moves": [], "analyzeTurns": [0, 1]}),
                json.dumps({"id": "game-b", "moves": [], "analyzeTurns": [0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    log = tmp_path / "analysis.log"

    first_report = run_katago_analysis_streaming(
        katago_bin=engine,
        model=tmp_path / "model.bin.gz",
        config=tmp_path / "analysis.cfg",
        requests_path=requests,
        out_path=responses,
        log_path=log,
        max_inflight_positions=1,
    )
    assert first_report["processes_started"] == 1
    assert first_report["response_lines"] == 3
    assert log.read_text(encoding="utf-8").count("engine started") == 1

    resumed_report = run_katago_analysis_streaming(
        katago_bin=engine,
        model=tmp_path / "model.bin.gz",
        config=tmp_path / "analysis.cfg",
        requests_path=requests,
        out_path=responses,
        log_path=log,
        max_inflight_positions=1,
    )
    assert resumed_report["processes_started"] == 0
    assert resumed_report["resumed_responses"] == 3
    assert log.read_text(encoding="utf-8").count("engine started") == 1
