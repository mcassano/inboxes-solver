"""Find, rectify, and measure a puzzle grid in a photograph.

A photo of a phone screen differs from a screenshot in two ways that both
break transcription, and neither is about being "blurry":

- **Keystone.** The phone is never exactly parallel to the camera, so the
  grid is a trapezoid, not a rectangle. Measured on a real capture, cell
  width drifted 58px to 66px across the grid -- 14%. Any attempt to map a
  number to a cell by proportional position is then off by up to half a cell
  near the edges, which is exactly the off-by-one row/column errors vision
  models make on these photos.
- **Contrast.** The thin grid lines come out as faint grey on washed-out
  cream, under a lighting gradient. Measured line contrast on that same
  capture was a standard deviation of 2.7 brightness levels; after
  flat-fielding and adaptive thresholding it was 17.9.

So we do the geometry here, in OpenCV, rather than asking a model to infer
it: find the grid's quadrilateral, warp it to a true rectangle, flatten the
lighting, and recover the row/column counts from the lattice pitch. The
model is then only asked to read digits out of a clean, square, correctly
sized grid -- and can be told the dimensions instead of guessing them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The grid must occupy at least this fraction of the image to be considered.
MIN_GRID_AREA_FRACTION = 0.10

# Plausible aspect ratios for the grid's bounding box.
MIN_ASPECT, MAX_ASPECT = 0.3, 2.2

# Lattice pitch is searched between these pixel bounds.
MIN_PITCH, MAX_PITCH = 18, 220

# Row and column pitch must agree within this fraction for cells to count as
# square, which is what lets us trust the derived dimensions.
MAX_PITCH_DISAGREEMENT = 0.12

# Autocorrelation strength below which the lattice reading isn't trusted.
MIN_AUTOCORR = 0.25

# Derived row/col counts must be within this of a whole number.
MAX_COUNT_ROUNDING_ERROR = 0.25

MIN_ROWS_COLS, MAX_ROWS_COLS = 4, 20


@dataclass
class RectifiedGrid:
    """A grid photo straightened out, with its dimensions measured."""

    image: np.ndarray  # binarized, perspective-corrected, cropped to the grid
    rows: int
    cols: int


def flat_field(gray: np.ndarray, sigma: float = 25) -> np.ndarray:
    """Divide out slowly-varying illumination so thin lines survive."""
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, background, scale=255)


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )


def find_grid_quad(ink: np.ndarray) -> np.ndarray | None:
    """The four corners of the largest plausible grid-shaped contour."""
    height, width = ink.shape
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < MIN_GRID_AREA_FRACTION * height * width:
            continue
        if not (MIN_ASPECT < w / h < MAX_ASPECT):
            continue
        if best is None or w * h > best[0]:
            best = (w * h, contour)
    if best is None:
        return None

    contour = best[1]
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))
    return None


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    total = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    return np.array(
        [
            quad[np.argmin(total)],  # top-left: smallest x+y
            quad[np.argmin(diff)],  # top-right: smallest y-x
            quad[np.argmax(total)],  # bottom-right
            quad[np.argmax(diff)],  # bottom-left
        ],
        dtype=np.float32,
    )


def warp_to_rect(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct the quadrilateral into a true rectangle."""
    top_left, top_right, bottom_right, bottom_left = quad
    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if width < 2 or height < 2:
        raise ValueError("degenerate grid quadrilateral")
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, target), (width, height))


def lattice_pitch(profile: np.ndarray) -> tuple[int, float] | None:
    """Dominant repeat distance in a projection profile, via autocorrelation.

    Robust to the faint, broken grid lines in a photo, where looking for
    individual lines finds only a handful of the strongest ones.
    """
    centered = profile - profile.mean()
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    best_lag, best_value = None, -1.0
    upper = min(MAX_PITCH, len(correlation) - 1)
    for lag in range(MIN_PITCH, upper):
        value = correlation[lag]
        if value > best_value and value > correlation[lag - 1] and value >= correlation[lag + 1]:
            best_lag, best_value = lag, float(value)
    if best_lag is None:
        return None
    return best_lag, best_value


def measure_dimensions(binary: np.ndarray) -> tuple[int, int] | None:
    """Recover (rows, cols) from the lattice, or None if not confident."""
    ink = (binary < 128).astype(float)
    height, width = ink.shape

    columns = lattice_pitch(ink.sum(axis=0))
    rows = lattice_pitch(ink.sum(axis=1))
    if columns is None or rows is None:
        return None
    col_pitch, col_strength = columns
    row_pitch, row_strength = rows
    if min(col_strength, row_strength) < MIN_AUTOCORR:
        return None

    # Cells are square, so the two pitches must agree; if they don't, we've
    # locked onto something other than the grid.
    if abs(col_pitch - row_pitch) / max(col_pitch, row_pitch) > MAX_PITCH_DISAGREEMENT:
        return None

    exact_cols, exact_rows = width / col_pitch, height / row_pitch
    n_cols, n_rows = round(exact_cols), round(exact_rows)
    if abs(exact_cols - n_cols) > MAX_COUNT_ROUNDING_ERROR:
        return None
    if abs(exact_rows - n_rows) > MAX_COUNT_ROUNDING_ERROR:
        return None
    if not (MIN_ROWS_COLS <= n_rows <= MAX_ROWS_COLS and MIN_ROWS_COLS <= n_cols <= MAX_ROWS_COLS):
        return None
    return n_rows, n_cols


def rectify(bgr: np.ndarray) -> RectifiedGrid | None:
    """Straighten and measure the puzzle grid in a photo.

    Returns None whenever any step is not confident, so callers can fall back
    to using the original image unchanged.
    """
    if bgr is None or bgr.size == 0:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    flat = flat_field(gray)
    quad = find_grid_quad(_ink_mask(flat))
    if quad is None:
        return None
    try:
        warped = warp_to_rect(flat, quad)
    except ValueError:
        return None

    binary = binarize(warped)
    dimensions = measure_dimensions(binary)
    if dimensions is None:
        return None
    rows, cols = dimensions
    return RectifiedGrid(image=binary, rows=rows, cols=cols)
