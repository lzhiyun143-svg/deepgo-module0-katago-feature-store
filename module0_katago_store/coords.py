from __future__ import annotations

BOARD_SIZE = 19
GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"
PASS_INDEX = BOARD_SIZE * BOARD_SIZE


def coord_to_rc(coord: str, board_size: int = BOARD_SIZE) -> tuple[int, int] | None:
    """Convert a KataGo/GTP coordinate such as Q16 to zero-based row, col."""
    c = coord.strip().upper()
    if c == "PASS":
        return None
    if len(c) < 2:
        raise ValueError(f"Invalid coordinate: {coord!r}")
    col_letter = c[0]
    if col_letter == "I" or col_letter not in GO_COLUMNS[:board_size]:
        raise ValueError(f"Invalid Go column: {coord!r}")
    try:
        row_from_bottom = int(c[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid Go row: {coord!r}") from exc
    if not 1 <= row_from_bottom <= board_size:
        raise ValueError(f"Row out of range for board size {board_size}: {coord!r}")
    col = GO_COLUMNS.index(col_letter)
    row = board_size - row_from_bottom
    return row, col


def rc_to_coord(row: int, col: int, board_size: int = BOARD_SIZE) -> str:
    if not 0 <= row < board_size or not 0 <= col < board_size:
        raise ValueError(f"row/col out of range: {(row, col)}")
    return f"{GO_COLUMNS[col]}{board_size - row}"


def coord_to_flat(coord: str, board_size: int = BOARD_SIZE) -> int:
    rc = coord_to_rc(coord, board_size)
    if rc is None:
        return board_size * board_size
    row, col = rc
    return row * board_size + col


def flat_to_coord(index: int, board_size: int = BOARD_SIZE) -> str:
    if index == board_size * board_size:
        return "pass"
    if not 0 <= index < board_size * board_size:
        raise ValueError(f"flat index out of range: {index}")
    return rc_to_coord(index // board_size, index % board_size, board_size)


def policy_to_map(policy: list[float], board_size: int = BOARD_SIZE) -> tuple[list[list[float]], float]:
    expected = board_size * board_size + 1
    if len(policy) != expected:
        raise ValueError(f"Expected policy length {expected}, got {len(policy)}")
    board = policy[: board_size * board_size]
    mapped = [board[i : i + board_size] for i in range(0, len(board), board_size)]
    return mapped, float(policy[-1])

