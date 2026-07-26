from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import read_jsonl
from .schema import MissingFeatureError, NotFoundError, SchemaMismatchError


class FeatureStore:
    def __init__(self, root: str | Path, expected_schema_version: str = "1.0"):
        self.root = Path(root)
        self.expected_schema_version = expected_schema_version
        self.profile = self._load_json(self.root / "metadata" / "analysis_profile.json", default={})
        self.schema = self._load_json(self.root / "metadata" / "schema.json", default={"schema_version": "1.0"})
        actual = self.schema.get("schema_version", self.profile.get("schema_version", "1.0"))
        if actual != expected_schema_version:
            raise SchemaMismatchError(f"Expected schema {expected_schema_version}, got {actual}")
        self.index = {row["position_id"]: row for row in read_jsonl(self.root / "index" / "position_index.jsonl")}
        self.scalars = self._load_scalars()
        self._spatial_cache: dict[Path, np.lib.npyio.NpzFile] = {}

    def close(self) -> None:
        for handle in self._spatial_cache.values():
            handle.close()
        self._spatial_cache.clear()

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _load_json(self, path: Path, default: dict) -> dict:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_scalars(self) -> dict[str, dict]:
        rows = {}
        base = self.root / "normalized" / "scalars"
        if not base.exists():
            return rows
        for part in base.glob("*/*.jsonl"):
            for rec in read_jsonl(part):
                rows[rec["position_id"]] = rec
        return rows

    def describe(self) -> dict:
        splits: dict[str, int] = {}
        for row in self.index.values():
            splits[row["split"]] = splits.get(row["split"], 0) + 1
        return {
            "root": str(self.root),
            "profile_id": self.profile.get("profile_id"),
            "schema_version": self.schema.get("schema_version", self.profile.get("schema_version")),
            "positions": len(self.index),
            "splits": splits,
        }

    def get(self, position_id: str, fields: Iterable[str] | None = None) -> dict:
        if position_id not in self.index:
            raise NotFoundError(position_id)
        scalar = dict(self.scalars.get(position_id, {}))
        idx = self.index[position_id]
        scalar.setdefault("position_id", position_id)
        scalar.setdefault("game_id", idx["game_id"])
        scalar.setdefault("turn_number", idx["turn_number"])
        scalar.setdefault("split", idx["split"])

        requested = set(fields or [])
        if not requested:
            requested = set(scalar) | {"policy_map", "ownership_map", "legal_mask"}

        out = {k: v for k, v in scalar.items() if k in requested or not fields}
        spatial_fields = {"policy_map", "ownership_map", "legal_mask"} & requested
        if spatial_fields:
            spatial = self._load_spatial(idx)
            row = int(idx["spatial_row"])
            if "policy_map" in spatial_fields:
                out["policy_map"] = spatial["policy"][row]
            if "ownership_map" in spatial_fields:
                out["ownership_map"] = spatial["ownership"][row]
            if "legal_mask" in spatial_fields:
                out["legal_mask"] = spatial["legal_mask"][row]

        missing = [field for field in requested if field not in out]
        if missing:
            raise MissingFeatureError(f"{position_id}: {missing}")
        return out

    def get_many(self, position_ids: Iterable[str], fields: Iterable[str] | None = None) -> list[dict]:
        return [self.get(pid, fields=fields) for pid in position_ids]

    def get_game(self, game_id: str, turns: Iterable[int] | None = None, fields: Iterable[str] | None = None) -> list[dict]:
        turn_set = set(turns) if turns is not None else None
        rows = [
            row
            for row in self.index.values()
            if row["game_id"] == game_id and (turn_set is None or int(row["turn_number"]) in turn_set)
        ]
        rows.sort(key=lambda r: int(r["turn_number"]))
        return [self.get(row["position_id"], fields=fields) for row in rows]

    def _load_spatial(self, idx: dict) -> np.lib.npyio.NpzFile:
        path = self.root / "normalized" / "spatial" / idx["spatial_shard"]
        if path not in self._spatial_cache:
            self._spatial_cache[path] = np.load(path, allow_pickle=True)
        return self._spatial_cache[path]
