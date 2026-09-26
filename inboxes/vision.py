"""Decode a screenshot of an Economist Inboxes puzzle into the standardized
JSON format (see inboxes.puzzle) using a vision-capable model on OpenRouter.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
from PIL import Image, ImageChops

from . import grid_detect, openrouter
from .puzzle import Clue, Puzzle
from .solver import Unsolvable, has_unique_solution

# Cap on the longer edge of the image sent to the model, in pixels, after
# cropping and upscaling (see _prepare_image). Keeps the base64 payload
# bounded even for very large source screenshots.
MAX_IMAGE_DIMENSION = 2200

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

# Tried, in order, after DEFAULT_MODEL (or whatever model the caller asked
# for) if it fails to produce a puzzle that passes every check in
# _decode_once. Vision models each have their own, largely consistent
# mis-transcription habits on this kind of dense numeric grid -- e.g. one
# model may reliably drop a blank row, while another reliably shifts a
# couple of clues in a specific row one column to the right -- so re-asking
# the *same* model rarely fixes it (it tends to make the same mistake
# again), but a different model often reads the trouble spot correctly.
FALLBACK_MODELS = ["google/gemini-2.5-pro", "openai/gpt-4o"]

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

As a final check, add up every number you transcribed. The clue rectangles \
tile the grid exactly, so that total ALWAYS equals (rows) x (columns). If it \
doesn't, your row or column count is wrong (the numbers themselves are \
rarely the problem) -- recount and fix it before answering.
"""

# Used instead of the trailing paragraph above when the grid has already been
# located and measured by inboxes.grid_detect: the image is then rectified and
# cropped to the grid exactly, and the dimensions are known, so asking the
# model to count rows only invites the drift this is meant to avoid.
MEASURED_GRID_PROMPT = """\
You transcribe a Shikaku ("Inboxes") puzzle grid from an image: a grid of \
cells, some holding a number.

This image has been cropped to EXACTLY the grid and perspective-corrected, so \
the grid fills it edge to edge in equal-sized square cells. It has EXACTLY \
{rows} rows and EXACTLY {cols} columns. Those counts were measured from the \
image itself and are correct -- do not recount them, and do not output any \
other shape.

Read every one of the {rows}x{cols} cells and report its number, or null if \
the cell is empty. Row r spans the vertical fraction r/{rows} to \
(r+1)/{rows}; column c spans the horizontal fraction c/{cols} to \
(c+1)/{cols}.

Output ONLY a JSON object (no markdown fences, no commentary) of this shape, \
with exactly {rows} inner arrays of exactly {cols} entries each:

{{"grid": [[null, 6, null, ...], ...]}}
"""


def _prepare_image(path: str | Path) -> str:
    """Load a screenshot, crop it to its non-white content, upscale it for
    clarity, and return it as a data: URL.

    Puzzle screenshots (especially from a phone) are often mostly blank
    margin around a grid that ends up occupying a small, low-resolution
    fraction of the image -- e.g. a status bar and a large blank area below
    the grid. Trimming that away and enlarging what's left measurably
    improves transcription accuracy without needing any puzzle-specific
    grid-line detection: it's a plain whitespace autocrop.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        flattened = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        flattened.paste(rgba, mask=rgba.split()[-1])
        img = flattened

    background = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, background).getbbox()
    if bbox is not None:
        pad = 8
        left, top, right, bottom = bbox
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img.width, right + pad)
        bottom = min(img.height, bottom + pad)
        img = img.crop((left, top, right, bottom))

    return _to_data_url(img)


def _to_data_url(img: Image.Image) -> str:
    """Upscale for legibility (capped) and encode as a PNG data URL."""
    scale = max(1.0, min(2.0, MAX_IMAGE_DIMENSION / max(img.size)))
    if scale > 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _prepare_for_decode(path: str | Path) -> tuple[str, tuple[int, int] | None]:
    """Return (data URL, measured (rows, cols) or None) for an image.

    Prefers a rectified, measured grid (see inboxes.grid_detect), which is
    what makes photographs usable: it removes the keystone distortion that
    otherwise shifts clues into neighbouring cells, and it establishes the
    grid dimensions by measurement instead of leaving the model to count
    faint lines. Falls back to the plain autocrop whenever the grid can't be
    found confidently.
    """
    grid = grid_detect.rectify(cv2.imread(str(path)))
    if grid is None:
        return _prepare_image(path), None
    return _to_data_url(Image.fromarray(grid.image).convert("RGB")), (grid.rows, grid.cols)


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


def _decode_once(
    model: str, data_url: str, dimensions: tuple[int, int] | None = None
) -> Puzzle:
    """One single-shot decode attempt, fully checked. Raises on any defect."""
    if dimensions is None:
        prompt = SYSTEM_PROMPT
    else:
        rows, cols = dimensions
        prompt = MEASURED_GRID_PROMPT.format(rows=rows, cols=cols)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this puzzle grid as JSON."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    content = openrouter.chat(model, messages, timeout=120, max_tokens=4096)
    data = openrouter.extract_json(content)
    puzzle = _grid_to_puzzle(data)
    puzzle.validate()
    # A clue shifted to the wrong cell can leave the value sum untouched, so
    # validate() alone won't catch it. Run it through the real solver: a
    # genuine Inboxes puzzle always has exactly one tiling, so anything
    # unsolvable, or ambiguous, is almost certainly a mis-transcription.
    if not has_unique_solution(puzzle):
        raise ValueError("puzzle has more than one valid solution (likely a mis-transcribed clue)")
    return puzzle


def decode_screenshot(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    attempts_per_model: int = 3,
) -> Puzzle:
    """Send a puzzle screenshot to a vision model on OpenRouter and return a Puzzle.

    `model` is tried first, followed by FALLBACK_MODELS, each for up to
    `attempts_per_model` independent single-shot attempts; the first
    fully-checked result wins. Falling back to a different model (rather
    than just re-asking the same one) matters because these mistakes tend to
    be a consistent habit of a given model on a given image, not random
    noise -- re-sampling the same model usually reproduces the same error.
    """
    data_url, dimensions = _prepare_for_decode(image_path)
    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error: Exception | None = None
    for candidate_model in models:
        for _ in range(attempts_per_model):
            try:
                return _decode_once(candidate_model, data_url, dimensions)
            except (ValueError, KeyError, TypeError, Unsolvable, openrouter.OpenRouterError) as exc:
                last_error = exc
    raise VisionDecodeError(
        f"could not decode a valid puzzle from {image_path} after trying "
        f"{', '.join(models)}: {last_error}"
    ) from last_error
