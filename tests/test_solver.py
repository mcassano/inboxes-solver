import time

import pytest

from inboxes.puzzle import Clue, Puzzle
from inboxes.solver import ShikakuSolver, Unsolvable, has_unique_solution, solve_puzzle, verify_solution

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


def test_regression_colliding_forced_singles_are_not_placed():
    # Regression test for a real puzzle that used to trigger overlapping
    # rectangles: two clues could each look like a "forced single" against
    # the grid state at the start of a propagation pass, but their sole
    # candidates overlapped each other. The propagator placed both anyway
    # without re-checking the grid in between, corrupting the tiling.
    puzzle = Puzzle(
        rows=11,
        cols=9,
        clues=[
            Clue(row=0, col=1, value=6),
            Clue(row=0, col=4, value=6),
            Clue(row=0, col=8, value=6),
            Clue(row=1, col=3, value=8),
            Clue(row=2, col=6, value=2),
            Clue(row=2, col=8, value=3),
            Clue(row=3, col=0, value=2),
            Clue(row=3, col=4, value=6),
            Clue(row=5, col=1, value=6),
            Clue(row=5, col=8, value=2),
            Clue(row=6, col=0, value=5),
            Clue(row=6, col=2, value=2),
            Clue(row=6, col=5, value=4),
            Clue(row=6, col=6, value=8),
            Clue(row=7, col=3, value=4),
            Clue(row=8, col=1, value=2),
            Clue(row=8, col=2, value=2),
            Clue(row=8, col=4, value=2),
            Clue(row=8, col=5, value=2),
            Clue(row=8, col=7, value=4),
            Clue(row=9, col=3, value=4),
            Clue(row=10, col=1, value=2),
            Clue(row=10, col=4, value=3),
            Clue(row=10, col=6, value=4),
            Clue(row=10, col=7, value=4),
        ],
    )
    solved = solve_puzzle(puzzle)
    verify_solution(solved)  # raises ValueError on overlap, area mismatch, etc.


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


def test_count_solutions_counts_a_genuinely_ambiguous_puzzle():
    # Diagonal 2x2 clues: tileable as two horizontal or two vertical strips.
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=0, col=0, value=2), Clue(row=1, col=1, value=2)])
    assert ShikakuSolver(puzzle).count_solutions(limit=5) == 2


def test_count_solutions_respects_limit():
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=0, col=0, value=2), Clue(row=1, col=1, value=2)])
    assert ShikakuSolver(puzzle).count_solutions(limit=1) == 1


def test_has_unique_solution_true_for_forced_puzzle():
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
    assert has_unique_solution(puzzle) is True


def test_has_unique_solution_false_for_ambiguous_puzzle():
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=0, col=0, value=2), Clue(row=1, col=1, value=2)])
    assert has_unique_solution(puzzle) is False


def test_has_unique_solution_raises_for_unsolvable_puzzle():
    puzzle = Puzzle(
        rows=2,
        cols=3,
        clues=[Clue(row=0, col=0, value=3), Clue(row=0, col=2, value=3)],
    )
    with pytest.raises(Unsolvable):
        has_unique_solution(puzzle)
