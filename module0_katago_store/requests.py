from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .io import read_jsonl, write_jsonl_atomic


def build_analysis_requests(store_root: Path, out_path: Path, profile: dict) -> list[dict]:
    games = {g["game_id"]: g for g in read_jsonl(store_root / "metadata" / "dataset_manifest.jsonl")}
    turns_by_game: dict[str, list[int]] = defaultdict(list)
    for pos in read_jsonl(store_root / "metadata" / "position_manifest.jsonl"):
        turns_by_game[pos["game_id"]].append(int(pos["turn_number"]))

    requests: list[dict] = []
    for game_id, turns in sorted(turns_by_game.items()):
        game = games[game_id]
        query = {
            "id": game_id,
            "moves": game["moves"],
            "rules": game.get("rules") or profile.get("rules_fallback", "chinese"),
            "komi": game.get("komi", profile.get("komi_fallback", 7.5)),
            "boardXSize": 19,
            "boardYSize": 19,
            "analyzeTurns": sorted(turns),
            "maxVisits": profile["max_visits"],
            "analysisPVLen": profile["analysis_pv_len"],
            "includePolicy": profile["include_policy"],
            "includeOwnership": profile["include_ownership"],
        }
        requests.append(query)

    write_jsonl_atomic(out_path, requests)
    return requests

