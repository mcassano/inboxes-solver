"""Optional alternative solver that asks a text model on OpenRouter to solve
the puzzle directly, instead of using the deterministic ShikakuSolver in
inboxes.solver. Useful as a cross-check; the deterministic solver is the
recommended default since it is exact and much faster.
"""

from __future__ import annotations

import json

from . import openrouter
from .puzzle import Puzzle
from .solver import verify_solution

DEFAULT_MODEL = "openai/gpt-5"

SYSTEM_PROMPT = """\
You solve Shikaku logic puzzles (the Economist calls this game "Inboxes"). \
You will be given a JSON description of the grid:

{"rows": R, "cols": C, "clues": [{"row": r, "col": c, "value": n}, ...]}

Partition the R x C grid into non-overlapping rectangles, one per clue, so \
that each rectangle contains exactly its own clue cell and no other clue \
cell, has area exactly equal to that clue's value, and every cell in the \
grid is covered by exactly one rectangle.

Respond with ONLY a JSON object (no markdown fences, no commentary) in this \
exact shape, using 0-indexed, inclusive coordinates:

{"rows": R, "cols": C, "clues": [
  {"row": r, "col": c, "value": n, "r0": r0, "c0": c0, "r1": r1, "c1": c1},
  ...
]}
"""


def llm_solve(puzzle: Puzzle, model: str = DEFAULT_MODEL) -> Puzzle:
    """Ask an LLM to solve the puzzle directly. Raises ValueError if its
    answer doesn't validate as a correct tiling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(puzzle.to_dict())},
    ]
    content = openrouter.chat(model, messages, timeout=180)
    data = openrouter.extract_json(content)
    solved = Puzzle.from_dict(data)
    verify_solution(solved)
    return solved
