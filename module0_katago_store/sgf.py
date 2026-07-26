from __future__ import annotations

import re
from pathlib import Path

SGF_PROP_RE = re.compile(r"([A-Za-z]+)((?:\[[^\]]*\])+)")
SGF_VALUE_RE = re.compile(r"\[([^\]]*)\]")


def _values(blob: str) -> list[str]:
    return SGF_VALUE_RE.findall(blob)


def sgf_point_to_gtp(point: str, board_size: int = 19) -> str:
    if point == "":
        return "pass"
    if len(point) != 2:
        raise ValueError(f"Unsupported SGF point: {point!r}")
    col = ord(point[0]) - ord("a")
    row_from_top = ord(point[1]) - ord("a")
    if not 0 <= col < board_size or not 0 <= row_from_top < board_size:
        raise ValueError(f"SGF point out of range: {point!r}")
    letters = "ABCDEFGHJKLMNOPQRST"
    return f"{letters[col]}{board_size - row_from_top}"


def parse_sgf(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    props: dict[str, list[str]] = {}
    for key, blob in SGF_PROP_RE.findall(text):
        props.setdefault(key.upper(), []).extend(_values(blob))

    board_size = int(props.get("SZ", ["19"])[0] or 19)
    komi_raw = props.get("KM", ["7.5"])[0]
    try:
        komi = float(komi_raw)
    except ValueError:
        komi = 7.5

    moves: list[tuple[str, str]] = []
    for color, blob in re.findall(r";([BW])(\[[^\]]*\])", text):
        move = _values(blob)[0]
        moves.append((color, sgf_point_to_gtp(move, board_size)))

    return {
        "board_size": board_size,
        "rules": props.get("RU", ["chinese"])[0] or "chinese",
        "komi": komi,
        "players": {"B": props.get("PB", [""])[0], "W": props.get("PW", [""])[0]},
        "result": props.get("RE", [""])[0],
        "moves": moves,
    }

