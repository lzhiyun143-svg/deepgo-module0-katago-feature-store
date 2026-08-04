from __future__ import annotations

import json
from pathlib import Path

from module1_environment.core.transition import TransitionRecord


class TransitionWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: TransitionRecord) -> None:
        self._handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TransitionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
