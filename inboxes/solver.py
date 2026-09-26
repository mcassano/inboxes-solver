"""Deterministic Shikaku solver for Inboxes puzzles.

Standard technique for this puzzle family:
  1. For every clue, enumerate every rectangle of the right area that
     contains the clue's cell and no other clue's cell.
  2. Repeatedly prune candidates that now overlap already-placed
     rectangles, and force any clue left with exactly one candidate
     ("naked single" propagation).
  3. When propagation stalls, branch on the clue with the fewest
     remaining candidates and recurse (classic backtracking).

Economist Inboxes puzzles are designed to have a unique solution and are
small enough (a few dozen clues on an 8x12 grid) that this converges
quickly without needing a full Algorithm X / DLX implementation.
"""

from __future__ import annotations

from .puzzle import Puzzle

Rect = tuple[int, int, int, int]  # r0, c0, r1, c1 (inclusive)


class Unsolvable(Exception):
    pass


class ShikakuSolver:
    def __init__(self, puzzle: Puzzle):
        puzzle.validate()
        self.puzzle = puzzle
        self.rows = puzzle.rows
        self.cols = puzzle.cols
        self.candidates: list[list[Rect]] = self._generate_candidates()
        for idx, opts in enumerate(self.candidates):
            if not opts:
                c = puzzle.clues[idx]
                raise Unsolvable(
                    f"clue {c.value} at ({c.row}, {c.col}) has no valid rectangle placement"
                )

    def _generate_candidates(self) -> list[list[Rect]]:
        clue_positions = {(c.row, c.col) for c in self.puzzle.clues}
        all_candidates = []
        for c in self.puzzle.clues:
            n = c.value
            options: list[Rect] = []
            for h in range(1, n + 1):
                if n % h:
                    continue
                w = n // h
                if h > self.rows or w > self.cols:
                    continue
                r0_lo = max(0, c.row - h + 1)
                r0_hi = min(c.row, self.rows - h)
                c0_lo = max(0, c.col - w + 1)
                c0_hi = min(c.col, self.cols - w)
                for r0 in range(r0_lo, r0_hi + 1):
                    r1 = r0 + h - 1
                    for c0 in range(c0_lo, c0_hi + 1):
                        c1 = c0 + w - 1
                        if self._contains_other_clue(r0, c0, r1, c1, clue_positions, c.row, c.col):
                            continue
                        options.append((r0, c0, r1, c1))
            all_candidates.append(options)
        return all_candidates

    @staticmethod
    def _contains_other_clue(r0, c0, r1, c1, clue_positions, own_row, own_col) -> bool:
        for (rr, cc) in clue_positions:
            if (rr, cc) == (own_row, own_col):
                continue
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                return True
        return False

    @staticmethod
    def _fits(grid, rect: Rect) -> bool:
        r0, c0, r1, c1 = rect
        for r in range(r0, r1 + 1):
            row = grid[r]
            for c in range(c0, c1 + 1):
                if row[c] != -1:
                    return False
        return True

    @staticmethod
    def _place(grid, rect: Rect, idx: int) -> None:
        r0, c0, r1, c1 = rect
        for r in range(r0, r1 + 1):
            row = grid[r]
            for c in range(c0, c1 + 1):
                row[c] = idx

    def _propagate(self, grid, remaining: dict[int, list[Rect]]):
        """Prune impossible candidates and force naked singles until stable.

        Returns (grid, remaining) or None on conflict.
        """
        grid = [row[:] for row in grid]
        remaining = {k: list(v) for k, v in remaining.items()}
        changed = True
        while changed:
            changed = False
            for idx in list(remaining.keys()):
                opts = remaining[idx]
                filtered = [r for r in opts if self._fits(grid, r)]
                if not filtered:
                    return None
                if len(filtered) != len(opts):
                    changed = True
                remaining[idx] = filtered
            for idx in list(remaining.keys()):
                opts = remaining[idx]
                if len(opts) == 1:
                    self._place(grid, opts[0], idx)
                    del remaining[idx]
                    changed = True
        return grid, remaining

    def _search(self, grid, remaining: dict[int, list[Rect]]):
        result = self._propagate(grid, remaining)
        if result is None:
            return None
        grid, remaining = result
        if not remaining:
            return grid
        idx = min(remaining, key=lambda i: len(remaining[i]))
        for rect in remaining[idx]:
            new_grid = [row[:] for row in grid]
            self._place(new_grid, rect, idx)
            new_remaining = dict(remaining)
            del new_remaining[idx]
            solved = self._search(new_grid, new_remaining)
            if solved is not None:
                return solved
        return None

    def solve(self) -> Puzzle:
        grid = [[-1] * self.cols for _ in range(self.rows)]
        remaining = {i: list(opts) for i, opts in enumerate(self.candidates)}
        solved_grid = self._search(grid, remaining)
        if solved_grid is None:
            raise Unsolvable("no valid tiling found for this puzzle")
        return self._grid_to_solution(solved_grid)

    def _grid_to_solution(self, grid) -> Puzzle:
        bounds = {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = grid[r][c]
                if idx not in bounds:
                    bounds[idx] = [r, c, r, c]
                else:
                    b = bounds[idx]
                    b[0] = min(b[0], r)
                    b[1] = min(b[1], c)
                    b[2] = max(b[2], r)
                    b[3] = max(b[3], c)
        solved_clues = []
        for idx, c in enumerate(self.puzzle.clues):
            r0, c0, r1, c1 = bounds[idx]
            solved_clues.append(
                type(c)(row=c.row, col=c.col, value=c.value, r0=r0, c0=c0, r1=r1, c1=c1)
            )
        return Puzzle(rows=self.rows, cols=self.cols, clues=solved_clues)


def solve_puzzle(puzzle: Puzzle) -> Puzzle:
    return ShikakuSolver(puzzle).solve()


def verify_solution(puzzle: Puzzle) -> None:
    """Raise ValueError if the solved puzzle is not a valid tiling."""
    grid = [[-1] * puzzle.cols for _ in range(puzzle.rows)]
    for idx, c in enumerate(puzzle.clues):
        if not c.solved:
            raise ValueError(f"clue {c} is not solved")
        if c.area() != c.value:
            raise ValueError(f"clue {c} rectangle area {c.area()} != value {c.value}")
        if not (c.r0 <= c.row <= c.r1 and c.c0 <= c.col <= c.c1):
            raise ValueError(f"clue {c} is not inside its own rectangle")
        for r in range(c.r0, c.r1 + 1):
            for cc in range(c.c0, c.c1 + 1):
                if grid[r][cc] != -1:
                    raise ValueError(f"rectangles overlap at ({r}, {cc})")
                grid[r][cc] = idx
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            if grid[r][c] == -1:
                raise ValueError(f"cell ({r}, {c}) is not covered by any rectangle")
