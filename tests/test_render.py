from PIL import Image

from inboxes.puzzle import Clue, Puzzle
from inboxes.render import CELL_SIZE, MARGIN, render_puzzle
from inboxes.solver import solve_puzzle


def test_render_unsolved_puzzle_writes_correctly_sized_png(tmp_path):
    puzzle = Puzzle(rows=2, cols=3, clues=[Clue(row=0, col=0, value=6)])
    out = tmp_path / "puzzle.png"
    render_puzzle(puzzle, out)
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (MARGIN * 2 + 3 * CELL_SIZE, MARGIN * 2 + 2 * CELL_SIZE)


def test_render_solved_puzzle(tmp_path):
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
    out = tmp_path / "solution.png"
    render_puzzle(solved, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_creates_missing_parent_directories(tmp_path):
    puzzle = Puzzle(rows=1, cols=1, clues=[Clue(row=0, col=0, value=1)])
    out = tmp_path / "nested" / "dir" / "puzzle.png"
    render_puzzle(puzzle, out)
    assert out.exists()
