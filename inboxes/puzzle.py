"""Standardized JSON format for an Inboxes (Shikaku) puzzle.

Puzzle format
-------------
{
  "rows": 8,
  "cols": 12,
  "clues": [
    {"row": 0, "col": 3, "value": 4},
    ...
  ]
}

Rows/cols are 0-indexed. Every clue's "value" is the area (in cells) of the
rectangle it must anchor.

Solution format (superset of the puzzle format)
------------------------------------------------
Same as above, plus each clue gains "r0", "c0", "r1", "c1": the inclusive
top-left / bottom-right corners of the rectangle assigned to that clue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Clue:
    row: int
    col: int
    value: int
    # Filled in once solved.
    r0: int | None = None
    c0: int | None = None
    r1: int | None = None
    c1: int | None = None

    @property
    def solved(self) -> bool:
        return self.r0 is not None

    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)


@dataclass
class Puzzle:
    rows: int
    cols: int
    clues: list[Clue] = field(default_factory=list)

    def total_clue_area(self) -> int:
        return sum(c.value for c in self.clues)

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows/cols must be positive")
        seen = set()
        for c in self.clues:
            if not (0 <= c.row < self.rows and 0 <= c.col < self.cols):
                raise ValueError(f"clue {c} is outside the {self.rows}x{self.cols} grid")
            if (c.row, c.col) in seen:
                raise ValueError(f"two clues at the same cell ({c.row}, {c.col})")
            seen.add((c.row, c.col))
            if c.value <= 0:
                raise ValueError(f"clue {c} has non-positive value")
        total = self.total_clue_area()
        if total != self.rows * self.cols:
            raise ValueError(
                f"clue areas sum to {total}, but grid has {self.rows * self.cols} cells"
            )

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "clues": [
                {k: v for k, v in asdict(c).items() if v is not None} for c in self.clues
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "Puzzle":
        clues = [Clue(**c) for c in data["clues"]]
        return cls(rows=data["rows"], cols=data["cols"], clues=clues)

    @classmethod
    def load(cls, path: str | Path) -> "Puzzle":
        return cls.from_dict(json.loads(Path(path).read_text()))
