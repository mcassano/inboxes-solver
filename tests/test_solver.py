import time

import pytest

from inboxes.puzzle import Clue, Puzzle
from inboxes.solver import Unsolvable, solve_puzzle, verify_solution

from .helpers import random_puzzle


def test_solves_four_quadrant_grid():
    # A 4x4 grid split into four 2x2 blocks. Each clue's only geometrically
    # possible placement (that avoids the other clues) is its own block, so
    # this has one, forced, unique solution.
    puzzle = Puzzle(
        rows=4,
        cols=4,
        clues=[
            Clue(row=0, col=0, value=4),
            Clue(row=0, col=2, value=4),
            Clue(row=2, col=0, value=4),
            Clue(row=2, col=2, value=4),
        ],
    )
    solved = solve_puzzle(puzzle)
    verify_solution(solved)
    bounds = {(c.row, c.col): (c.r0, c.c0, c.r1, c.c1) for c in solved.clues}
    assert bounds[(0, 0)] == (0, 0, 1, 1)
    assert bounds[(0, 2)] == (0, 2, 1, 3)
    assert bounds[(2, 0)] == (2, 0, 3, 1)
    assert bounds[(2, 2)] == (2, 2, 3, 3)


def test_single_clue_covers_whole_grid():
    puzzle = Puzzle(rows=3, cols=3, clues=[Clue(row=1, col=1, value=9)])
    solved = solve_puzzle(puzzle)
    verify_solution(solved)
    clue = solved.clues[0]
    assert (clue.r0, clue.c0, clue.r1, clue.c1) == (0, 0, 2, 2)


def test_unsolvable_raises():
    # The only geometrically possible rectangle for the value-3 clue at
    # (0, 0) in a 2x3 grid is the whole top row, which swallows the other
    # clue -- so there is no valid placement for it at all.
    puzzle = Puzzle(
        rows=2,
        cols=3,
        clues=[Clue(row=0, col=0, value=3), Clue(row=0, col=2, value=3)],
    )
    with pytest.raises(Unsolvable):
        solve_puzzle(puzzle)


@pytest.mark.parametrize("seed", range(20))
def test_random_small_puzzles_are_solved_correctly(seed):
    puzzle, _ = random_puzzle(rows=5, cols=5, seed=seed)
    solved = solve_puzzle(puzzle)
    verify_solution(solved)  # raises on any invalid tiling


@pytest.mark.parametrize("seed", range(5))
def test_random_full_size_puzzles_are_solved_correctly(seed):
    puzzle, _ = random_puzzle(rows=8, cols=12, seed=seed + 100)
    solved = solve_puzzle(puzzle)
    verify_solution(solved)


def test_solves_full_size_puzzle_quickly():
    puzzle, _ = random_puzzle(rows=8, cols=12, seed=42)
    start = time.monotonic()
    solved = solve_puzzle(puzzle)
    elapsed = time.monotonic() - start
    verify_solution(solved)
    assert elapsed < 10


def test_verify_solution_detects_overlap():
    puzzle = Puzzle(
        rows=1,
        cols=2,
        clues=[
            Clue(row=0, col=0, value=2, r0=0, c0=0, r1=0, c1=1),
            Clue(row=0, col=1, value=2, r0=0, c0=0, r1=0, c1=1),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        verify_solution(puzzle)


def test_verify_solution_detects_area_mismatch():
    puzzle = Puzzle(
        rows=1,
        cols=2,
        clues=[Clue(row=0, col=0, value=1, r0=0, c0=0, r1=0, c1=1)],
    )
    with pytest.raises(ValueError, match="!="):
        verify_solution(puzzle)


def test_verify_solution_detects_uncovered_cell():
    puzzle = Puzzle(
        rows=1,
        cols=2,
        clues=[Clue(row=0, col=0, value=1, r0=0, c0=0, r1=0, c1=0)],
    )
    with pytest.raises(ValueError, match="not covered"):
        verify_solution(puzzle)
