"""Shared test helpers: a random valid Shikaku tiling generator.

Recursively splits an R x C grid into rectangles (a guillotine cut kd-tree),
then places one clue per rectangle. The result is guaranteed solvable (the
generated tiling is a witness), which makes it useful for exercising the
solver without needing hand-authored puzzles.
"""

from __future__ import annotations

import random

from inboxes.puzzle import Clue, Puzzle


def _split(r0, c0, r1, c1, rng, min_leaf_area=1):
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = height * width

    stop = area <= min_leaf_area * 2 or rng.random() < 0.3
    can_split_rows = height > 1
    can_split_cols = width > 1

    if stop or not (can_split_rows or can_split_cols):
        return [(r0, c0, r1, c1)]

    split_rows = can_split_rows and (not can_split_cols or rng.random() < 0.5)
    if split_rows:
        cut = rng.randint(r0, r1 - 1)
        return _split(r0, c0, cut, c1, rng, min_leaf_area) + _split(
            cut + 1, c0, r1, c1, rng, min_leaf_area
        )
    else:
        cut = rng.randint(c0, c1 - 1)
        return _split(r0, c0, r1, cut, rng, min_leaf_area) + _split(
            r0, cut + 1, r1, c1, rng, min_leaf_area
        )


def random_puzzle(rows: int, cols: int, seed: int) -> tuple[Puzzle, list[tuple[int, int, int, int]]]:
    """Return (puzzle, rectangles) for a randomly generated, solvable grid."""
    rng = random.Random(seed)
    rectangles = _split(0, 0, rows - 1, cols - 1, rng)
    clues = []
    for r0, c0, r1, c1 in rectangles:
        row = rng.randint(r0, r1)
        col = rng.randint(c0, c1)
        value = (r1 - r0 + 1) * (c1 - c0 + 1)
        clues.append(Clue(row=row, col=col, value=value))
    puzzle = Puzzle(rows=rows, cols=cols, clues=clues)
    puzzle.validate()
    return puzzle, rectangles
