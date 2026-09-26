import json

import pytest

from inboxes.puzzle import Clue, Puzzle


def make_valid_puzzle() -> Puzzle:
    return Puzzle(
        rows=2,
        cols=2,
        clues=[Clue(row=0, col=0, value=2), Clue(row=0, col=1, value=2)],
    )


def test_validate_accepts_matching_area():
    make_valid_puzzle().validate()  # should not raise


def test_validate_rejects_area_mismatch():
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=0, col=0, value=3)])
    with pytest.raises(ValueError, match="clue areas sum"):
        puzzle.validate()


def test_validate_rejects_duplicate_clue_cell():
    puzzle = Puzzle(
        rows=2,
        cols=2,
        clues=[Clue(row=0, col=0, value=2), Clue(row=0, col=0, value=2)],
    )
    with pytest.raises(ValueError, match="same cell"):
        puzzle.validate()


def test_validate_rejects_out_of_bounds_clue():
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=5, col=0, value=4)])
    with pytest.raises(ValueError, match="outside"):
        puzzle.validate()


def test_validate_rejects_non_positive_value():
    puzzle = Puzzle(rows=2, cols=2, clues=[Clue(row=0, col=0, value=0), Clue(row=0, col=1, value=4)])
    with pytest.raises(ValueError, match="non-positive"):
        puzzle.validate()


def test_to_dict_omits_unsolved_bounds():
    data = make_valid_puzzle().to_dict()
    for clue in data["clues"]:
        assert "r0" not in clue and "c0" not in clue and "r1" not in clue and "c1" not in clue


def test_to_dict_includes_solved_bounds():
    puzzle = Puzzle(
        rows=1,
        cols=2,
        clues=[Clue(row=0, col=0, value=2, r0=0, c0=0, r1=0, c1=1)],
    )
    data = puzzle.to_dict()
    assert data["clues"][0]["r1"] == 0
    assert data["clues"][0]["c1"] == 1


def test_round_trip_dict():
    original = make_valid_puzzle()
    restored = Puzzle.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.rows == original.rows
    assert restored.cols == original.cols
    assert [(c.row, c.col, c.value) for c in restored.clues] == [
        (c.row, c.col, c.value) for c in original.clues
    ]


def test_save_and_load_round_trip(tmp_path):
    original = make_valid_puzzle()
    path = tmp_path / "puzzle.json"
    original.save(path)
    restored = Puzzle.load(path)
    assert restored.to_dict() == original.to_dict()


def test_clue_area_and_solved():
    clue = Clue(row=0, col=0, value=4)
    assert not clue.solved
    clue.r0, clue.c0, clue.r1, clue.c1 = 0, 0, 1, 1
    assert clue.solved
    assert clue.area() == 4
