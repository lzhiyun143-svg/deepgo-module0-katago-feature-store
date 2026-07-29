from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .coords import BOARD_SIZE, coord_to_flat, policy_to_map
from .ids import position_id
from .io import read_jsonl, write_jsonl_atomic


def _root_info(resp: dict) -> dict:
    return resp.get("rootInfo") or resp.get("rootInfoDuringSearch") or {}


def _policy_metrics(policy: np.ndarray, human_index: int) -> dict:
    legal = policy[: BOARD_SIZE * BOARD_SIZE].copy()
    legal[legal < 0] = 0.0
    total = float(legal.sum())
    if total > 0:
        probs = legal / total
    else:
        probs = legal
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    ranked = np.argsort(-probs)
    rank = int(np.where(ranked == human_index)[0][0] + 1) if 0 <= human_index < BOARD_SIZE * BOARD_SIZE else None
    human_policy = float(probs[human_index]) if rank is not None else None
    gap = float(probs[ranked[0]] - probs[ranked[1]]) if len(ranked) > 1 else None
    return {
        "policy_entropy": entropy,
        "effective_candidate_count": float(math.exp(entropy)),
        "human_move_policy": human_policy,
        "human_move_rank_policy": rank,
        "policy_gap_top1_top2": gap,
    }


def _candidate_records(resp: dict, pos: dict, top_n: int) -> list[dict]:
    rows = []
    for rank, item in enumerate((resp.get("moveInfos") or [])[:top_n], 1):
        rows.append(
            {
                "position_id": pos["position_id"],
                "game_id": pos["game_id"],
                "turn_number": pos["turn_number"],
                "candidate_rank": rank,
                "move": item.get("move"),
                "prior": item.get("prior"),
                "visits": item.get("visits"),
                "winrate": item.get("winrate"),
                "score_lead": item.get("scoreLead"),
                "utility": item.get("utility"),
                "pv": item.get("pv", []),
            }
        )
    return rows


def normalize_responses(
    store_root: Path,
    responses_path: Path,
    profile: dict,
    split: str | None = None,
    top_n: int = 10,
) -> dict:
    positions = {
        (p["game_id"], int(p["turn_number"])): p
        for p in read_jsonl(store_root / "metadata" / "position_manifest.jsonl")
    }
    scalars_by_split: dict[str, list[dict]] = defaultdict(list)
    candidates_by_split: dict[str, list[dict]] = defaultdict(list)
    spatial_by_split: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    index_rows: list[dict] = []
    missing_positions = 0

    for resp in read_jsonl(responses_path):
        if resp.get("isDuringSearch"):
            continue
        game_id = str(resp.get("id", "")).split("#", 1)[0]
        turn = int(resp.get("turnNumber", resp.get("turn", 0)))
        pos = positions.get((game_id, turn))
        if pos is None:
            missing_positions += 1
            continue
        pos_split = pos["split"]
        if split and pos_split != split:
            continue
        pid = position_id(game_id, turn)
        root = _root_info(resp)
        human_index = coord_to_flat(pos["human_move"])
        policy = np.asarray(resp.get("policy", [-1.0] * (BOARD_SIZE * BOARD_SIZE) + [0.0]), dtype=np.float32)
        policy_map, policy_pass = policy_to_map(policy.tolist())
        metrics = _policy_metrics(policy, human_index)
        ownership = resp.get("ownership")
        if ownership is None:
            ownership_map = np.full((BOARD_SIZE, BOARD_SIZE), np.nan, dtype=np.float32)
            has_ownership = False
        else:
            ownership_map = np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
            has_ownership = True

        scalar = {
            "position_id": pid,
            "game_id": game_id,
            "turn_number": turn,
            "split": pos_split,
            "player_to_move": pos["player_to_move"],
            "human_move": pos["human_move"],
            "human_move_index": human_index,
            "policy_pass": policy_pass,
            "root_winrate": root.get("winrate"),
            "root_score_lead": root.get("scoreLead"),
            "root_utility": root.get("utility"),
            "raw_winrate": root.get("rawWinrate"),
            "visits": root.get("visits"),
            "this_hash": root.get("thisHash"),
            "human_move_eval_status": "covered" if any(m.get("move") == pos["human_move"] for m in resp.get("moveInfos", [])) else "not_covered",
            "profile_id": profile["profile_id"],
            "schema_version": profile.get("schema_version", "1.0"),
            **metrics,
        }
        scalars_by_split[pos_split].append(scalar)
        candidates_by_split[pos_split].extend(_candidate_records(resp, pos, top_n))
        row_offset = len(spatial_by_split[pos_split]["position_ids"])
        spatial_by_split[pos_split]["position_ids"].append(pid)
        spatial_by_split[pos_split]["policy"].append(np.asarray(policy_map, dtype=np.float32))
        spatial_by_split[pos_split]["ownership"].append(ownership_map)
        spatial_by_split[pos_split]["legal_mask"].append((policy[: BOARD_SIZE * BOARD_SIZE] >= 0).reshape(BOARD_SIZE, BOARD_SIZE))
        index_rows.append(
            {
                "position_id": pid,
                "game_id": game_id,
                "turn_number": turn,
                "split": pos_split,
                "spatial_shard": f"{pos_split}/shard-00000.npz",
                "spatial_row": row_offset,
                "has_ownership": has_ownership,
            }
        )

    for pos_split, rows in scalars_by_split.items():
        write_jsonl_atomic(store_root / "normalized" / "scalars" / pos_split / "part-00000.jsonl", rows)
        write_jsonl_atomic(store_root / "normalized" / "candidates" / pos_split / "part-00000.jsonl", candidates_by_split[pos_split])
        shard = store_root / "normalized" / "spatial" / pos_split / "shard-00000.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        arrays = spatial_by_split[pos_split]
        np.savez_compressed(
            shard,
            position_ids=np.asarray(arrays["position_ids"], dtype=object),
            policy=np.asarray(arrays["policy"], dtype=np.float32),
            ownership=np.asarray(arrays["ownership"], dtype=np.float32),
            legal_mask=np.asarray(arrays["legal_mask"], dtype=bool),
        )
    write_jsonl_atomic(store_root / "index" / "position_index.jsonl", index_rows)
    return {"positions_written": len(index_rows), "unknown_responses": missing_positions}

