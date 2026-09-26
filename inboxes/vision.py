"""Decode a screenshot of an Economist Inboxes puzzle into the standardized
JSON format (see inboxes.puzzle) using a vision-capable model on OpenRouter.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from . import openrouter
from .puzzle import Clue, Puzzle

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

# The model is asked for a dense, row-major grid (one array per row, one
# entry per column, null for empty cells) rather than a sparse list of
# clues. A sparse list requires the model to track a running row/column
# index in its head, and it's easy for it to silently lose count across a
# row that has no clues in it at all. A dense grid can't be "skipped" the
# same way: every row of the image must produce one array in the output,
# even if that array is all nulls, which is a much easier invariant for the
# model to hold onto than a mental counter.
SYSTEM_PROMPT = """\
You read screenshots of the Economist's "Inboxes" puzzle (a Shikaku-style \
logic puzzle: a rectangular grid where some cells contain a number, and the \
grid must be partitioned into rectangles, each containing exactly one \
number equal to that rectangle's area in cells).

Transcribe the ENTIRE grid, top to bottom and left to right, as a dense 2D \
array: one inner array per row of the grid, one entry per column of that \
row, using the clue number where a cell has one and JSON null where it does \
not. Every row must produce an inner array, including rows that have no \
clues in them at all -- an empty row is `[null, null, ..., null]`, not a \
skipped row. Every inner array must have the same length (the number of \
columns).

Output ONLY a JSON object (no markdown fences, no commentary) of this shape:

{
  "grid": [
    [null, 6, null, null, null, 12, null, null, null],
    [null, null, 2, null, null, null, null, null, 3],
    [null, null, null, null, null, null, null, null, null],
    ...
  ]
}

Work row by row down the image. Before answering, count the grid lines to \
confirm the number of inner arrays matches the number of rows in the image \
and each inner array's length matches the number of columns -- do not infer \
the grid size from where the numbers are.
"""


def _image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class VisionDecodeError(RuntimeError):
    pass


def _grid_to_puzzle(data: dict) -> Puzzle:
    grid = data["grid"]
    if not isinstance(grid, list) or not grid:
        raise ValueError("'grid' must be a non-empty list of rows")
    cols = len(grid[0])
    clues = []
    for r, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(
                f"row {r} has {len(row) if isinstance(row, list) else 'invalid'} "
                f"entries, expected {cols} (every row must have the same length)"
            )
        for c, value in enumerate(row):
            if value is None:
                continue
            clues.append(Clue(row=r, col=c, value=int(value)))
    return Puzzle(rows=len(grid), cols=cols, clues=clues)


def decode_screenshot(image_path: str | Path, model: str = DEFAULT_MODEL, max_attempts: int = 3) -> Puzzle:
    """Send a puzzle screenshot to a vision model on OpenRouter and return a Puzzle.

    The most common mistake vision models make on this task is mis-counting
    the grid's rows or columns (especially ones with no clues in them at
    all); asking for a dense grid instead of a sparse clue list (see
    SYSTEM_PROMPT) largely prevents that. As a second line of defence,
    Puzzle.validate() catches anything that still slips through -- a real
    Inboxes puzzle always has clue values that sum to exactly rows * cols --
    so on a validation failure we report the error back to the model and
    give it another shot, up to max_attempts tries, before giving up.
    """
    data_url = _image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this puzzle grid as JSON."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        content = openrouter.chat(model, messages, timeout=120)
        try:
            data = openrouter.extract_json(content)
            puzzle = _grid_to_puzzle(data)
            puzzle.validate()
            return puzzle
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That is invalid: {exc}. Re-examine the screenshot, recount the grid "
                        "lines and clue values row by row, and answer again with the corrected "
                        "grid as JSON only."
                    ),
                }
            )
    raise VisionDecodeError(
        f"could not decode a valid puzzle from {image_path} after {max_attempts} attempts: {last_error}"
    ) from last_error
