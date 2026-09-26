"""Decode a screenshot of an Economist Inboxes puzzle into the standardized
JSON format (see inboxes.puzzle) using a vision-capable model on OpenRouter.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from . import openrouter
from .puzzle import Puzzle

DEFAULT_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """\
You read screenshots of the Economist's "Inboxes" puzzle (a Shikaku-style \
logic puzzle: a rectangular grid where some cells contain a number, and the \
grid must be partitioned into rectangles, each containing exactly one \
number equal to that rectangle's area in cells).

Given a screenshot, output ONLY a JSON object (no markdown fences, no \
commentary) describing the grid using 0-indexed rows/columns counted from \
the top-left of the puzzle grid:

{
  "rows": <number of grid rows>,
  "cols": <number of grid columns>,
  "clues": [
    {"row": <0-indexed row>, "col": <0-indexed col>, "value": <the number shown>},
    ...
  ]
}

Include one entry in "clues" for every numbered cell visible in the \
screenshot, and no entry for empty cells. Double check every row and column \
of the grid before answering so no clue is missed or misplaced.
"""


def _image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def decode_screenshot(image_path: str | Path, model: str = DEFAULT_MODEL) -> Puzzle:
    """Send a puzzle screenshot to a vision model on OpenRouter and return a Puzzle."""
    data_url = _image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the puzzle grid from this screenshot as JSON."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    content = openrouter.chat(model, messages, timeout=120)
    data = openrouter.extract_json(content)
    puzzle = Puzzle.from_dict(data)
    puzzle.validate()
    return puzzle
