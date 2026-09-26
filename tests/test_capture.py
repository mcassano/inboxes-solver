import time

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from inboxes.capture import (  # noqa: E402
    CaptureError,
    SettleDetector,
    _crop_to_screen,
    capture_from_webcam,
    detect_screen_bbox,
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


def test_capture_from_webcam_ignores_good_frames_before_space_is_pressed(tmp_path, monkeypatch):
    board = _checkerboard(size=32)
    sharp_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    frames = [sharp_bgr.copy() for _ in range(30)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    # No key for the first 15 (already sharp+steady) frames -- if arming
    # didn't gate the detector, it would have settled well before this.
    key_sequence = iter([-1] * 15 + [ord(" ")])
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(key_sequence, -1))

    out_path = tmp_path / "capture.png"
    result = capture_from_webcam(
        out_path=out_path,
        show_preview=True,
        sharpness_threshold=10,
        motion_threshold=1000,
        settle_frames=3,
        require_arm_key=True,
    )

    assert result == out_path
    # 15 unarmed frames + the space-press frame + 3 more to build the streak.
    assert len(fake._frames) <= 30 - 15 - 1 - 3


def test_capture_from_webcam_never_settles_without_pressing_space(tmp_path, monkeypatch):
    board = _checkerboard(size=32)
    sharp_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    frames = [sharp_bgr.copy() for _ in range(20)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)  # space never pressed

    with pytest.raises(CaptureError, match="failed to read a frame"):
        capture_from_webcam(
            out_path=tmp_path / "capture.png",
            show_preview=True,
            sharpness_threshold=10,
            motion_threshold=1000,
            settle_frames=3,
        )


def test_capture_from_webcam_abort_key_works_before_arming(tmp_path, monkeypatch):
    fake = _FakeCapture([_solid_frame(128) for _ in range(5)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: ord("q"))

    with pytest.raises(CaptureError, match="aborted by user"):
        capture_from_webcam(out_path=tmp_path / "capture.png", show_preview=True)


def test_capture_from_webcam_timeout_only_counts_after_arming(tmp_path, monkeypatch):
    blank = _solid_frame(128)  # never sharp enough to settle once armed

    class _InfiniteFakeCapture(_FakeCapture):
        def read(self):
            return True, blank.copy()

    fake = _InfiniteFakeCapture(frames=[])
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    # Each waitKey call advances a fake clock by 1s; space is pressed on the
    # 5th call. If the 5 "unarmed" fake-seconds counted against the 2s
    # timeout, this would raise almost immediately after arming instead of
    # ~2-3 fake-seconds later.
    calls = {"n": 0}

    def fake_waitkey(_delay):
        calls["n"] += 1
        clock["t"] += 1.0
        return ord(" ") if calls["n"] == 5 else -1

    monkeypatch.setattr(cv2, "waitKey", fake_waitkey)

    with pytest.raises(CaptureError, match="no sharp, steady frame"):
        capture_from_webcam(
            out_path=tmp_path / "capture.png",
            show_preview=True,
            sharpness_threshold=1e9,  # impossible, so it never settles once armed
            timeout_seconds=2.0,
        )
    assert clock["t"] < 10.0


def _frame_with_bright_rect(size=(300, 400), rect=(80, 50, 320, 250), bg=30, fg=220):
    height, width = size
    frame = np.full((height, width, 3), bg, dtype=np.uint8)
    x0, y0, x1, y1 = rect
    frame[y0:y1, x0:x1] = fg
    return frame


class TestDetectScreenBbox:
    def test_finds_a_bright_rectangle_on_a_dark_background(self):
        frame = _frame_with_bright_rect()
        bbox = detect_screen_bbox(frame)
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert abs(x0 - 80) <= 5
        assert abs(y0 - 50) <= 5
        assert abs(x1 - 320) <= 5
        assert abs(y1 - 250) <= 5

    def test_returns_none_when_bright_region_is_too_small(self):
        # A tiny bright patch, well under MIN_SCREEN_AREA_FRACTION of the frame.
        frame = _frame_with_bright_rect(size=(400, 400), rect=(10, 10, 30, 30))
        assert detect_screen_bbox(frame) is None

    def test_returns_none_for_a_uniform_frame(self):
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert detect_screen_bbox(frame) is None


class TestCropToScreen:
    def test_crops_to_the_detected_screen(self):
        frame = _frame_with_bright_rect(size=(300, 400))
        cropped = _crop_to_screen(frame)
        assert cropped.shape[0] < 300
        assert cropped.shape[1] < 400
        # Should still be a sensible size, not degenerate.
        assert cropped.shape[0] > 150
        assert cropped.shape[1] > 150

    def test_falls_back_to_the_full_frame_when_nothing_detected(self):
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        cropped = _crop_to_screen(frame)
        assert cropped.shape == frame.shape

    def test_never_warns_about_touching_a_frame_edge(self, capsys):
        # This warning was removed: it fired on every real capture while the
        # puzzle was entirely in view, because glare on the phone's glass
        # widens the detected bright region. Whether the grid is complete is
        # now established downstream by grid_detect and the solver.
        frame = _frame_with_bright_rect(size=(200, 300), rect=(50, 0, 250, 200))
        _crop_to_screen(frame)
        assert capsys.readouterr().err == ""

    def test_crops_a_screen_that_reaches_a_frame_edge(self):
        frame = _frame_with_bright_rect(size=(200, 300), rect=(50, 0, 250, 200))
        cropped = _crop_to_screen(frame)
        assert cropped.shape[1] < frame.shape[1]


def test_capture_from_webcam_saves_a_frame_cropped_to_the_screen(tmp_path, monkeypatch):
    frame = _frame_with_bright_rect(size=(300, 400))
    frames = [frame.copy() for _ in range(5)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    out_path = tmp_path / "capture.png"
    capture_from_webcam(
        out_path=out_path,
        show_preview=False,
        sharpness_threshold=0,  # this synthetic frame has no real texture/focus
        motion_threshold=5,
        settle_frames=3,
    )

    saved = cv2.imread(str(out_path))
    assert saved.shape[0] < frame.shape[0]
    assert saved.shape[1] < frame.shape[1]


def test_capture_from_webcam_picks_sharpest_frame_from_burst(tmp_path, monkeypatch):
    # A single settled instant isn't necessarily the best available shot --
    # once settled, a short burst of extra frames is read and the sharpest
    # one is kept, even though it wasn't the frame that triggered settling.
    sharp = cv2.cvtColor(_checkerboard(size=32), cv2.COLOR_GRAY2BGR)
    dim = cv2.GaussianBlur(sharp, (9, 9), sigmaX=4)

    frames = [dim.copy() for _ in range(3)] + [dim.copy(), sharp.copy(), dim.copy(), dim.copy()]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)

    out_path = tmp_path / "capture.png"
    capture_from_webcam(
        out_path=out_path,
        show_preview=False,
        sharpness_threshold=1,  # low enough that the "dim" frames still settle
        motion_threshold=1000,
        settle_frames=3,
    )

    saved_gray = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2GRAY)
    dim_gray = cv2.cvtColor(dim, cv2.COLOR_BGR2GRAY)
    assert sharpness_score(saved_gray) > sharpness_score(dim_gray) * 2
