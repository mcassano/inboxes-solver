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
DEFAULT_SHARPNESS_THRESHOLD = 150.0

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


class CaptureError(RuntimeError):
    pass


def sharpness_score(gray: np.ndarray) -> float:
    """Higher = more in-focus. `gray` is a single-channel image array."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Higher = more movement between the two (same-shape) grayscale frames."""
    return float(np.mean(cv2.absdiff(prev_gray, gray)))


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
) -> Path:
    """Wait for a sharp, steady frame from the webcam, save it, return its path.

    Hold the puzzle's phone screen up to the camera; once `settle_frames`
    consecutive frames are all sharp (see sharpness_score) and still (see
    motion_score), the current frame is saved to `out_path` and returned.
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
    start = time.monotonic()
    last_log = start
    window = "inboxes: hold the puzzle steady..."

    try:
        while True:
            if time.monotonic() - start > timeout_seconds:
                raise CaptureError(
                    f"no sharp, steady frame found within {timeout_seconds:.0f}s "
                    "-- is the camera pointed at the puzzle?"
                )

            ok, frame = cap.read()
            if not ok or frame is None:
                raise CaptureError(f"failed to read a frame from camera index {camera_index}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = sharpness_score(gray)
            motion = motion_score(prev_gray, gray) if prev_gray is not None else float("inf")
            settled = detector.update(sharpness, motion)
            prev_gray = gray

            now = time.monotonic()
            if debug_log and now - last_log >= DEBUG_LOG_INTERVAL_SECONDS:
                last_log = now
                print(
                    f"[capture] t={now - start:4.1f}s  sharpness={sharpness:7.1f} "
                    f"(need >= {sharpness_threshold:.0f})  motion={motion:5.2f} "
                    f"(need <= {motion_threshold:.1f})  streak={detector.streak}/{settle_frames}",
                    file=sys.stderr,
                )

            if show_preview:
                border = (0, 200, 0) if settled else (0, 0, 200)
                display = frame.copy()
                cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), border, 8)
                cv2.putText(
                    display,
                    f"sharpness={sharpness:6.1f} (need >={sharpness_threshold:.0f})  "
                    f"motion={motion:5.2f} (need <={motion_threshold:.1f})  "
                    f"streak={detector.streak}/{settle_frames}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0) if settled else (0, 0, 200),
                    2,
                )
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    raise CaptureError("aborted by user")

            if settled:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), frame)
                return out_path
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
