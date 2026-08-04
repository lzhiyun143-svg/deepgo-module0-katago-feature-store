from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from module0_katago_store.coords import BOARD_SIZE, PASS_INDEX, flat_to_coord

BLACK = 1
WHITE = -1
EMPTY = 0


@dataclass(frozen=True)
class MoveResult:
    legal: bool
    action_index: int
    action_gtp: str
    color: int
    captured: tuple[tuple[int, int], ...] = ()
    reason: str | None = None


@dataclass
class GoState:
    board: np.ndarray = field(default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))
    to_play: int = BLACK
    turn_number: int = 0
    consecutive_passes: int = 0
    move_history: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def empty(cls, to_play: int = BLACK) -> "GoState":
        return cls(to_play=to_play)

    def copy(self) -> "GoState":
        return GoState(
            board=self.board.copy(),
            to_play=self.to_play,
            turn_number=self.turn_number,
            consecutive_passes=self.consecutive_passes,
            move_history=list(self.move_history),
        )

    @property
    def player_to_move_label(self) -> str:
        return "B" if self.to_play == BLACK else "W"

    def state_hash(self) -> str:
        payload = self.board.tobytes() + bytes([1 if self.to_play == BLACK else 2]) + self.turn_number.to_bytes(4, "big")
        return sha256(payload).hexdigest()[:16]

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                mask[row, col] = self.is_legal(row * BOARD_SIZE + col)
        return mask

    def is_legal(self, action_index: int) -> bool:
        if action_index == PASS_INDEX:
            return True
        if not 0 <= action_index < PASS_INDEX:
            return False
        row, col = divmod(action_index, BOARD_SIZE)
        if self.board[row, col] != EMPTY:
            return False
        trial = self.copy()
        result = trial.apply(action_index, mutate=True)
        return result.legal

    def apply(self, action_index: int, mutate: bool = True) -> MoveResult:
        target = self if mutate else self.copy()
        color = target.to_play
        if action_index == PASS_INDEX:
            target.consecutive_passes += 1
            target.move_history.append((color, action_index))
            target.turn_number += 1
            target.to_play *= -1
            return MoveResult(True, action_index, "pass", color)

        if not 0 <= action_index < PASS_INDEX:
            return MoveResult(False, action_index, str(action_index), color, reason="out_of_range")

        row, col = divmod(action_index, BOARD_SIZE)
        action_gtp = flat_to_coord(action_index)
        if target.board[row, col] != EMPTY:
            return MoveResult(False, action_index, action_gtp, color, reason="occupied")

        target.board[row, col] = color
        captured: list[tuple[int, int]] = []
        for nr, nc in _neighbors(row, col):
            if target.board[nr, nc] == -color:
                group = _group(target.board, nr, nc)
                if not _has_liberty(target.board, group):
                    captured.extend(group)
        for nr, nc in captured:
            target.board[nr, nc] = EMPTY

        own_group = _group(target.board, row, col)
        if not _has_liberty(target.board, own_group):
            target.board[row, col] = EMPTY
            for nr, nc in captured:
                target.board[nr, nc] = -color
            return MoveResult(False, action_index, action_gtp, color, reason="suicide")

        target.consecutive_passes = 0
        target.move_history.append((color, action_index))
        target.turn_number += 1
        target.to_play *= -1
        return MoveResult(True, action_index, action_gtp, color, captured=tuple(captured))

    def done(self, max_turns: int | None = None) -> bool:
        if self.consecutive_passes >= 2:
            return True
        return max_turns is not None and self.turn_number >= max_turns

    def board_tensor(self) -> np.ndarray:
        tensor = np.zeros((17, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tensor[0] = self.board == BLACK
        tensor[1] = self.board == WHITE
        tensor[2] = self.board == EMPTY
        tensor[3].fill(1.0 if self.to_play == BLACK else 0.0)
        tensor[4].fill(self.turn_number / 361.0)
        recent = self.move_history[-6:]
        for offset, (_, action) in enumerate(reversed(recent), start=5):
            if action != PASS_INDEX and offset < 11:
                row, col = divmod(action, BOARD_SIZE)
                tensor[offset, row, col] = 1.0
        return tensor


def _neighbors(row: int, col: int):
    if row > 0:
        yield row - 1, col
    if row + 1 < BOARD_SIZE:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < BOARD_SIZE:
        yield row, col + 1


def _group(board: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    color = board[row, col]
    stack = [(row, col)]
    seen = {(row, col)}
    out: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        out.append((r, c))
        for nr, nc in _neighbors(r, c):
            if (nr, nc) not in seen and board[nr, nc] == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return out


def _has_liberty(board: np.ndarray, group: list[tuple[int, int]]) -> bool:
    for row, col in group:
        for nr, nc in _neighbors(row, col):
            if board[nr, nc] == EMPTY:
                return True
    return False
