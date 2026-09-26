import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from inboxes.capture import (  # noqa: E402
    CaptureError,
    SettleDetector,
    capture_from_webcam,
    motion_score,
    sharpness_score,
)


def _checkerboard(size=64, square=8):
    board = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if (x // square + y // square) % 2 == 0:
                board[y : y + square, x : x + square] = 255
    return board


def test_sharpness_score_is_higher_for_a_crisp_image():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (9, 9), sigmaX=4)
    assert sharpness_score(sharp) > sharpness_score(blurred)


def test_motion_score_zero_for_identical_frames():
    frame = _checkerboard()
    assert motion_score(frame, frame) == 0.0


def test_motion_score_positive_for_different_frames():
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.full((32, 32), 255, dtype=np.uint8)
    assert motion_score(a, b) == pytest.approx(255.0)


class TestSettleDetector:
    def test_settles_after_enough_consecutive_good_frames(self):
        detector = SettleDetector(sharpness_threshold=100, motion_threshold=5, settle_frames=3)
        assert detector.update(200, 1) is False
        assert detector.update(200, 1) is False
        assert detector.update(200, 1) is True

    def test_a_single_bad_frame_resets_the_streak(self):
        detector = SettleDetector(sharpness_threshold=100, motion_threshold=5, settle_frames=3)
        assert detector.update(200, 1) is False
        assert detector.update(200, 1) is False
        assert detector.update(50, 1) is False  # too blurry -- resets
        assert detector.update(200, 1) is False
        assert detector.update(200, 1) is False
        assert detector.update(200, 1) is True

    def test_never_settles_on_an_all_bad_stream(self):
        detector = SettleDetector(sharpness_threshold=100, motion_threshold=5, settle_frames=3)
        results = [detector.update(10, 50) for _ in range(20)]
        assert not any(results)

    def test_high_motion_alone_blocks_settling(self):
        detector = SettleDetector(sharpness_threshold=100, motion_threshold=5, settle_frames=2)
        assert detector.update(500, 50) is False
        assert detector.update(500, 50) is False

    def test_is_good_boundary_is_inclusive(self):
        detector = SettleDetector(sharpness_threshold=100, motion_threshold=5, settle_frames=1)
        assert detector.is_good(100, 5) is True
        assert detector.is_good(99.9, 5) is False
        assert detector.is_good(100, 5.1) is False


class _FakeCapture:
    """Stands in for cv2.VideoCapture, yielding a scripted list of frames."""

    def __init__(self, frames, opened=True):
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, *_args):
        return True

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


def _solid_frame(value, size=32):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_capture_from_webcam_saves_the_settled_frame(tmp_path, monkeypatch):
    # Three identical (still, and sharp enough once checkerboarded) frames in
    # a row should settle; use a checkerboard pattern for real sharpness.
    board = _checkerboard(size=32)
    sharp_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    frames = [sharp_bgr.copy() for _ in range(5)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    out_path = tmp_path / "capture.png"
    result = capture_from_webcam(
        out_path=out_path,
        show_preview=False,
        sharpness_threshold=10,
        motion_threshold=5,  # frames are identical, so motion is 0
        settle_frames=3,
    )

    assert result == out_path
    assert out_path.exists()
    assert fake.released


def test_capture_from_webcam_raises_when_camera_will_not_open(tmp_path, monkeypatch):
    fake = _FakeCapture(frames=[], opened=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    with pytest.raises(CaptureError, match="could not open camera"):
        capture_from_webcam(out_path=tmp_path / "capture.png", show_preview=False)


def test_capture_from_webcam_raises_on_read_failure(tmp_path, monkeypatch):
    fake = _FakeCapture(frames=[])  # opens fine, but .read() immediately fails
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    with pytest.raises(CaptureError, match="failed to read a frame"):
        capture_from_webcam(out_path=tmp_path / "capture.png", show_preview=False)
    assert fake.released


def test_capture_from_webcam_times_out_if_never_settled(tmp_path, monkeypatch):
    # An endless supply of frames that never pass the sharpness bar.
    blank = _solid_frame(128)

    class _InfiniteFakeCapture(_FakeCapture):
        def read(self):
            return True, blank.copy()

    fake = _InfiniteFakeCapture(frames=[])
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    with pytest.raises(CaptureError, match="no sharp, steady frame"):
        capture_from_webcam(
            out_path=tmp_path / "capture.png",
            show_preview=False,
            sharpness_threshold=1e9,  # impossible to satisfy
            timeout_seconds=0.05,
        )
    assert fake.released
