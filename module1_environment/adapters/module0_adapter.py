from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from module0_katago_store import FeatureStore
from module0_katago_store.io import read_jsonl
from module0_katago_store.schema import MissingFeatureError, NotFoundError


PREDECISION_FIELDS = [
    "position_id",
    "game_id",
    "turn_number",
    "split",
    "player_to_move",
    "root_winrate",
    "root_score_lead",
    "root_utility",
    "policy_entropy",
    "policy_gap_top1_top2",
    "policy_map",
    "ownership_map",
    "legal_mask",
]


class Module0Adapter:
    def __init__(self, store_root: str | Path, expected_schema_version: str = "1.0"):
        self.store = FeatureStore(store_root, expected_schema_version=expected_schema_version)
        self._candidates = self._load_candidates()

    def close(self) -> None:
        self.store.close()

    def describe(self) -> dict:
        return self.store.describe()

    def get_features(self, position_id: str, fields: Iterable[str] | None = None) -> dict:
        return self.store.get(position_id, fields=fields)

    def get_predecision_features(self, position_id: str) -> dict:
        try:
            features = self.store.get(position_id, fields=PREDECISION_FIELDS)
        except MissingFeatureError:
            safe_fields = [f for f in PREDECISION_FIELDS if f not in {"ownership_map"}]
            features = self.store.get(position_id, fields=safe_fields)
        features["top_candidates"] = self.get_candidates(position_id)
        return features

    def get_candidates(self, position_id: str, top_n: int = 10) -> list[dict]:
        rows = self._candidates.get(position_id, [])
        return rows[:top_n]

    def best_action_index(self, position_id: str) -> int | None:
        candidates = self.get_candidates(position_id, top_n=1)
        if not candidates:
            return None
        move = candidates[0].get("move")
        if not move:
            return None
        from module0_katago_store.coords import coord_to_flat

        return coord_to_flat(move)

    def has_position(self, position_id: str) -> bool:
        try:
            self.store.get(position_id, fields=["position_id"])
            return True
        except NotFoundError:
            return False

    def _load_candidates(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        base = self.store.root / "normalized" / "candidates"
        if not base.exists():
            return out
        for path in sorted(base.glob("*/*.jsonl")):
            for row in read_jsonl(path):
                out[row["position_id"]].append(row)
        for rows in out.values():
            rows.sort(key=lambda r: int(r.get("candidate_rank", 10**9)))
        return out
