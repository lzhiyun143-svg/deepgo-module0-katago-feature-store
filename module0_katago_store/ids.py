from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_game_id(source_path: str, source_sha256: str, prefix: str = "kgs") -> str:
    payload = f"{source_path}\n{source_sha256}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{prefix}_{digest}"


def position_id(game_id: str, turn_number: int) -> str:
    if turn_number < 0:
        raise ValueError("turn_number must be non-negative")
    return f"{game_id}__t{turn_number:06d}"

