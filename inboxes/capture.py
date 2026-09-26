"""Capture a puzzle photo from a webcam: wait for a sharp, steady shot of a
phone screen held up to the camera, then save it.

Two layers, kept separate so the decision logic is testable without a real
camera:

- sharpness_score / motion_score / SettleDetector: pure, no I/O.
- capture_from_webcam: talks to the camera via OpenCV.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# How often (in seconds) to print the live sharpness/motion readout to
# stderr while waiting. This is independent of the on-screen preview text,
# which flickers by too fast at ~30fps to read a specific number off of --
# the terminal log is what you'd actually copy-paste to report a problem.
DEBUG_LOG_INTERVAL_SECONDS = 0.5

DEFAULT_CAMERA_INDEX = 0
DEFAULT_OUT_PATH = Path("webcam_capture.png")

# Variance of the Laplacian of the grayscale frame -- a standard, simple
# focus/blur metric. Higher means sharper (more high-frequency detail).
#
# This is a full-frame average, so it's diluted by whatever fraction of the
# frame isn't the puzzle itself (background, hand, phone bezel) -- it's not
# comparable to "textbook" Laplacian-variance thresholds tuned on a frame
# that's mostly in-focus subject. 150 (a generic guess) was never reachable
# in practice: measured live against a Studio Display webcam with a phone
# screen clearly in frame and readable, steady-state values sat around
# 9-15, vs. ~0 with nothing in frame. 8.0 sits comfortably below that
# steady-state range with headroom, while still well above "nothing/totally
# blurry". If a different camera or distance needs a different number, the
# live [capture] log printed while waiting shows the real numbers to
# recalibrate --sharpness-threshold against.
DEFAULT_SHARPNESS_THRESHOLD = 8.0

# Mean absolute pixel difference (0-255 scale) between consecutive grayscale
# frames. Lower means less motion between frames.
DEFAULT_MOTION_THRESHOLD = 3.0

# Number of consecutive frames that must all pass both thresholds before a
# shot is accepted. Requiring a run (not just one lucky frame) avoids
# capturing on a single sharp instant in the middle of a moving swing.
DEFAULT_SETTLE_FRAMES = 10

DEFAULT_TIMEOUT_SECONDS = 60.0

REQUESTED_FRAME_WIDTH = 1920
REQUESTED_FRAME_HEIGHT = 1080

# A phone screen showing this puzzle is much brighter than a typical room
# background, so Otsu's method (which picks a threshold that best splits the
# image into two brightness clusters) reliably isolates it without needing a
# fixed brightness number that would vary by lighting.
MIN_SCREEN_AREA_FRACTION = 0.05

# On a frame with no real bright/dark split (e.g. pointed at a blank wall),
# Otsu's threshold is degenerate and can classify the entire frame as one
# blob. That's not a useful "screen" detection -- cropping to it would be a
# no-op anyway -- so treat anything this large as no detection.
MAX_SCREEN_AREA_FRACTION = 0.92

# How close (as a fraction of that dimension) the detected screen region can
# get to a frame edge before we warn that it looks cropped.
EDGE_WARNING_MARGIN_FRACTION = 0.02


class CaptureError(RuntimeError):
    pass


def sharpness_score(gray: np.ndarray) -> float:
    """Higher = more in-focus. `gray` is a single-channel image array."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Higher = more movement between the two (same-shape) grayscale frames."""
    return float(np.mean(cv2.absdiff(prev_gray, gray)))


def detect_screen_bbox(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the bright phone-screen rectangle in a webcam frame, if any.

    Returns (x0, y0, x1, y1) pixel bounds, or None if nothing confidently
    screen-shaped was found (caller should fall back to using the whole
    frame). This matters because a webcam photo's background usually isn't a
    plain white page the way a screenshot's is -- the whitespace autocrop in
    inboxes.vision._prepare_image barely trims a photo like this, leaving the
    puzzle as a small, low-resolution fraction of what gets sent to the
    vision model.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    area = cv2.contourArea(largest)
    if area < frame_area * MIN_SCREEN_AREA_FRACTION or area > frame_area * MAX_SCREEN_AREA_FRACTION:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def _crop_to_screen(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop a captured frame to its detected phone screen, with padding.

    Warns on stderr (but still returns the crop) if the detected region
    touches a frame edge, since that usually means part of the screen was
    outside the camera's field of view -- e.g. the puzzle's last row cut off
    the bottom of the shot.
    """
    bbox = detect_screen_bbox(frame_bgr)
    if bbox is None:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    margin_x = round(width * EDGE_WARNING_MARGIN_FRACTION)
    margin_y = round(height * EDGE_WARNING_MARGIN_FRACTION)
    if x0 <= margin_x or y0 <= margin_y or x1 >= width - margin_x or y1 >= height - margin_y:
        print(
            "warning: the detected phone screen touches the edge of the camera's "
            "view -- part of the puzzle may be cropped off. Try holding it further "
            "back, or centered lower/higher, so the whole grid is visible with "
            "some margin around it.",
            file=sys.stderr,
        )

    pad_x = round((x1 - x0) * 0.03)
    pad_y = round((y1 - y0) * 0.03)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return frame_bgr[y0:y1, x0:x1]


class SettleDetector:
    """Tracks a running streak of "good" (sharp and still) frames.

    Feed it one (sharpness, motion) reading per frame via `update`; it
    returns True once `settle_frames` consecutive readings have all passed
    both thresholds. Any reading that fails either threshold resets the
    streak to zero.
    """

    def __init__(
        self,
        sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        settle_frames: int = DEFAULT_SETTLE_FRAMES,
    ):
        self.sharpness_threshold = sharpness_threshold
        self.motion_threshold = motion_threshold
        self.settle_frames = settle_frames
        self.streak = 0

    def is_good(self, sharpness: float, motion: float) -> bool:
        return sharpness >= self.sharpness_threshold and motion <= self.motion_threshold

    def update(self, sharpness: float, motion: float) -> bool:
        """Record one frame's readings; return True once settled."""
        if self.is_good(sharpness, motion):
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.settle_frames


def capture_from_webcam(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    out_path: str | Path = DEFAULT_OUT_PATH,
    show_preview: bool = True,
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    debug_log: bool = True,
    require_arm_key: bool = True,
) -> Path:
    """Wait for a sharp, steady frame from the webcam, save it, return its path.

    With `require_arm_key` (the default whenever there's a preview window),
    nothing can be captured until you press SPACE to "arm" it -- take as long
    as you need to get the phone into frame first. Only once armed does the
    quality gate start counting: `settle_frames` consecutive frames all sharp
    (see sharpness_score) and still (see motion_score) after that point.
    Requiring the gate to pass *after* arming, rather than capturing on the
    keypress itself, also avoids the keypress's own jostle ruining the shot.
    `timeout_seconds` only starts counting down once armed.

    Without a preview window (or with `require_arm_key=False`) there's no way
    to press anything, so it's armed from the very first frame -- this is the
    fully automatic mode used by tests and any non-interactive caller.
    """
    out_path = Path(out_path)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise CaptureError(
            f"could not open camera index {camera_index} -- check System Settings > "
            "Privacy & Security > Camera has granted access, or try a different "
            "--camera-index"
        )

    detector = SettleDetector(sharpness_threshold, motion_threshold, settle_frames)
    prev_gray: np.ndarray | None = None
    armed = not (show_preview and require_arm_key)
    start = time.monotonic()
    armed_at = start if armed else None
    last_log = start
    window = "inboxes: hold the puzzle steady..."

    if not armed:
        print("Get the puzzle in frame, then press SPACE (preview window focused) to arm...", file=sys.stderr)

    try:
        while True:
            if armed and time.monotonic() - armed_at > timeout_seconds:
                raise CaptureError(
                    f"no sharp, steady frame found within {timeout_seconds:.0f}s of arming "
                    "-- is the camera pointed at the puzzle?"
                )

            ok, frame = cap.read()
            if not ok or frame is None:
                raise CaptureError(f"failed to read a frame from camera index {camera_index}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = sharpness_score(gray)
            motion = motion_score(prev_gray, gray) if prev_gray is not None else float("inf")
            settled = detector.update(sharpness, motion) if armed else False
            prev_gray = gray

            now = time.monotonic()
            if debug_log and now - last_log >= DEBUG_LOG_INTERVAL_SECONDS:
                last_log = now
                state = f"streak={detector.streak}/{settle_frames}" if armed else "not armed (press SPACE)"
                print(
                    f"[capture] t={now - start:4.1f}s  sharpness={sharpness:7.1f} "
                    f"(need >= {sharpness_threshold:.0f})  motion={motion:5.2f} "
                    f"(need <= {motion_threshold:.1f})  {state}",
                    file=sys.stderr,
                )

            if show_preview:
                if not armed:
                    border, label = (0, 165, 255), "press SPACE to arm"
                elif settled:
                    border, label = (0, 200, 0), "captured!"
                else:
                    border, label = (0, 0, 200), f"streak={detector.streak}/{settle_frames}"
                display = frame.copy()
                cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), border, 8)
                cv2.putText(
                    display,
                    f"{label}   sharpness={sharpness:6.1f} (need >={sharpness_threshold:.0f})  "
                    f"motion={motion:5.2f} (need <={motion_threshold:.1f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    border,
                    2,
                )
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    raise CaptureError("aborted by user")
                if not armed and key == ord(" "):
                    armed = True
                    armed_at = time.monotonic()
                    print("Armed -- hold steady...", file=sys.stderr)

            if settled:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), _crop_to_screen(frame))
                return out_path
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
