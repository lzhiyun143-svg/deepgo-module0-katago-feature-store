from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from module0_katago_store.io import read_jsonl


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    game_id: str
    split: str
    moves: tuple[tuple[str, str], ...]
    start_turn: int = 0
    max_steps: int | None = None
    source_path: str | None = None
    feature_store_release: str = "v1.0.0"


class EpisodeManifest:
    def __init__(self, rows: list[dict], feature_store_release: str = "v1.0.0"):
        self.rows = rows
        self.feature_store_release = feature_store_release

    @classmethod
    def from_module0_store(cls, store_root: str | Path) -> "EpisodeManifest":
        root = Path(store_root)
        rows = list(read_jsonl(root / "metadata" / "dataset_manifest.jsonl"))
        return cls(rows, feature_store_release=root.name)

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int = 0, game_id: str | None = None, max_steps: int | None = None) -> EpisodeSpec:
        if game_id is not None:
            row = next(row for row in self.rows if row["game_id"] == game_id)
        else:
            row = self.rows[index]
        moves = tuple((color, move) for color, move in row.get("moves", []))
        return EpisodeSpec(
            episode_id=f"{row['game_id']}__ep000",
            game_id=row["game_id"],
            split=row.get("split", "train"),
            moves=moves,
            max_steps=max_steps,
            source_path=row.get("source_path"),
            feature_store_release=self.feature_store_release,
        )
