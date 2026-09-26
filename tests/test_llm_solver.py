import json

import pytest

from inboxes import openrouter
from inboxes.llm_solver import llm_solve
from inboxes.puzzle import Clue, Puzzle


def make_puzzle() -> Puzzle:
    return Puzzle(rows=1, cols=2, clues=[Clue(row=0, col=0, value=2)])


def test_llm_solve_accepts_correct_solution(monkeypatch):
    solved_json = {
        "rows": 1,
        "cols": 2,
        "clues": [{"row": 0, "col": 0, "value": 2, "r0": 0, "c0": 0, "r1": 0, "c1": 1}],
    }

    def fake_chat(model, messages, timeout=180, temperature=0):
        sent = json.loads(messages[1]["content"])
        assert sent == make_puzzle().to_dict()
        return json.dumps(solved_json)

    monkeypatch.setattr(openrouter, "chat", fake_chat)
    solved = llm_solve(make_puzzle())
    assert solved.clues[0].r1 == 0
    assert solved.clues[0].c1 == 1


def test_llm_solve_rejects_invalid_tiling(monkeypatch):
    # Area of the returned rectangle (1) doesn't match the clue value (2).
    bad_json = {
        "rows": 1,
        "cols": 2,
        "clues": [{"row": 0, "col": 0, "value": 2, "r0": 0, "c0": 0, "r1": 0, "c1": 0}],
    }
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: json.dumps(bad_json))
    with pytest.raises(ValueError):
        llm_solve(make_puzzle())
