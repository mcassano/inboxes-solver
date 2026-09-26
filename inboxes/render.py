"""Render an Inboxes puzzle (solved or not) to a PNG image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .puzzle import Puzzle

CELL_SIZE = 48
MARGIN = 2
GRID_COLOR = (60, 60, 60)
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
RECT_BORDER_COLOR = (20, 20, 20)
RECT_BORDER_WIDTH = 3

# A small, muted palette cycled across solved rectangles so neighbors are
# visually distinguishable without implying any meaning by color.
PALETTE = [
    (255, 235, 205),
    (205, 235, 255),
    (220, 255, 220),
    (255, 220, 235),
    (240, 230, 255),
    (255, 250, 205),
    (220, 245, 245),
    (250, 225, 210),
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_puzzle(puzzle: Puzzle, path: str | Path) -> None:
    """Draw the grid and clue numbers, plus solved rectangles if present."""
    width = MARGIN * 2 + puzzle.cols * CELL_SIZE
    height = MARGIN * 2 + puzzle.rows * CELL_SIZE
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _load_font(int(CELL_SIZE * 0.45))

    def cell_box(r, c):
        x0 = MARGIN + c * CELL_SIZE
        y0 = MARGIN + r * CELL_SIZE
        return x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE

    solved = bool(puzzle.clues) and all(c.solved for c in puzzle.clues)

    if solved:
        for i, c in enumerate(puzzle.clues):
            x0, y0, _, _ = cell_box(c.r0, c.c0)
            _, _, x1, y1 = cell_box(c.r1, c.c1)
            draw.rectangle([x0, y0, x1, y1], fill=PALETTE[i % len(PALETTE)])

    for r in range(puzzle.rows + 1):
        y = MARGIN + r * CELL_SIZE
        draw.line([(MARGIN, y), (MARGIN + puzzle.cols * CELL_SIZE, y)], fill=GRID_COLOR)
    for c in range(puzzle.cols + 1):
        x = MARGIN + c * CELL_SIZE
        draw.line([(x, MARGIN), (x, MARGIN + puzzle.rows * CELL_SIZE)], fill=GRID_COLOR)

    if solved:
        for c in puzzle.clues:
            x0, y0, _, _ = cell_box(c.r0, c.c0)
            _, _, x1, y1 = cell_box(c.r1, c.c1)
            draw.rectangle([x0, y0, x1, y1], outline=RECT_BORDER_COLOR, width=RECT_BORDER_WIDTH)

    for c in puzzle.clues:
        x0, y0, x1, y1 = cell_box(c.row, c.col)
        text = str(c.value)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + (CELL_SIZE - tw) / 2 - bbox[0], y0 + (CELL_SIZE - th) / 2 - bbox[1]),
            text,
            fill=TEXT_COLOR,
            font=font,
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
