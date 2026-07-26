from __future__ import annotations

from pathlib import Path

from .ids import position_id, sha256_file, stable_game_id
from .io import write_jsonl_atomic
from .sgf import parse_sgf


def split_for_index(idx: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> str:
    bucket = idx % 10
    if bucket < int(train_ratio * 10):
        return "train"
    if bucket < int((train_ratio + valid_ratio) * 10):
        return "valid"
    return "test"


def build_manifests(
    sgf_dir: Path,
    store_root: Path,
    dataset_version: str = "kgs4d_v1",
    max_games: int | None = None,
) -> tuple[list[dict], list[dict]]:
    sgf_paths = sorted(p for p in sgf_dir.rglob("*.sgf") if p.is_file())
    if max_games:
        sgf_paths = sgf_paths[:max_games]

    games: list[dict] = []
    positions: list[dict] = []
    for idx, path in enumerate(sgf_paths):
        rel = path.relative_to(sgf_dir).as_posix()
        source_sha = sha256_file(path)
        game_id = stable_game_id(rel, source_sha)
        split = split_for_index(idx)
        try:
            parsed = parse_sgf(path)
            status = "success"
            error = None
        except Exception as exc:
            parsed = {"moves": [], "board_size": 19, "rules": "chinese", "komi": 7.5, "players": {}, "result": ""}
            status = "failed"
            error = str(exc)

        game = {
            "game_id": game_id,
            "source_path": rel,
            "source_sha256": source_sha,
            "dataset_version": dataset_version,
            "split": split,
            "board_size": parsed["board_size"],
            "rules": parsed["rules"],
            "komi": parsed["komi"],
            "players": parsed.get("players", {}),
            "result": parsed.get("result", ""),
            "moves_count": len(parsed["moves"]),
            "moves": parsed["moves"],
            "parse_status": status,
            "parse_error": error,
        }
        games.append(game)

        if status != "success" or parsed["board_size"] != 19:
            continue
        for turn in range(len(parsed["moves"])):
            human_color, human_move = parsed["moves"][turn]
            positions.append(
                {
                    "position_id": position_id(game_id, turn),
                    "game_id": game_id,
                    "turn_number": turn,
                    "split": split,
                    "player_to_move": human_color,
                    "human_move": human_move,
                    "board_size": parsed["board_size"],
                    "rules": parsed["rules"],
                    "komi": parsed["komi"],
                }
            )

    metadata = store_root / "metadata"
    write_jsonl_atomic(metadata / "dataset_manifest.jsonl", games)
    write_jsonl_atomic(metadata / "position_manifest.jsonl", positions)
    return games, positions

