import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from inboxes.grid_detect import (  # noqa: E402
    find_grid_quad,
    lattice_pitch,
    measure_dimensions,
    rectify,
    warp_to_rect,
)


def synthetic_grid(rows=11, cols=9, cell=40, margin=30, line=2, keystone=0.0, contrast=255):
    """A puzzle-like grid on a light background, optionally keystoned.

    `keystone` shifts the top edge inward by that fraction of the width, the
    way a tilted phone photographs.
    """
    h, w = rows * cell + 2 * margin, cols * cell + 2 * margin
    img = np.full((h, w, 3), 255, np.uint8)
    ink = 255 - contrast
    for r in range(rows + 1):
        y = margin + r * cell
        cv2.line(img, (margin, y), (margin + cols * cell, y), (ink,) * 3, line)
    for c in range(cols + 1):
        x = margin + c * cell
        cv2.line(img, (x, margin), (x, margin + rows * cell), (ink,) * 3, line)
    if keystone:
        dx = w * keystone
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[dx, 0], [w - 1 - dx, 0], [w - 1, h - 1], [0, h - 1]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderValue=(255, 255, 255))
    return img


class TestLatticePitch:
    def test_finds_the_repeat_distance_of_a_regular_comb(self):
        profile = np.zeros(400)
        profile[::25] = 1.0
        found = lattice_pitch(profile)
        assert found is not None
        pitch, strength = found
        assert pitch == pytest.approx(25, abs=1)
        assert strength > 0.5

    def test_returns_none_for_a_flat_profile(self):
        assert lattice_pitch(np.zeros(300)) is None


class TestMeasureDimensions:
    def test_measures_a_clean_grid(self):
        img = synthetic_grid(rows=11, cols=9, cell=40, margin=0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert measure_dimensions(gray) == (11, 9)

    def test_measures_a_different_shape(self):
        img = synthetic_grid(rows=8, cols=12, cell=45, margin=0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert measure_dimensions(gray) == (8, 12)

    def test_rejects_an_image_with_no_lattice(self):
        noise = np.full((300, 300), 255, np.uint8)
        assert measure_dimensions(noise) is None


class TestFindGridQuad:
    def test_finds_four_corners_of_a_grid(self):
        quad = find_grid_quad(
            cv2.adaptiveThreshold(
                cv2.cvtColor(synthetic_grid(), cv2.COLOR_BGR2GRAY),
                255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
            )
        )
        assert quad is not None
        assert quad.shape == (4, 2)
        # Ordered top-left, top-right, bottom-right, bottom-left.
        tl, tr, br, bl = quad
        assert tl[0] < tr[0] and bl[0] < br[0]
        assert tl[1] < bl[1] and tr[1] < br[1]

    def test_returns_none_when_there_is_no_grid(self):
        blank = np.zeros((200, 200), np.uint8)
        assert find_grid_quad(blank) is None


class TestWarpToRect:
    def test_straightens_a_keystoned_quad(self):
        img = synthetic_grid(keystone=0.08)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
        quad = find_grid_quad(ink)
        assert quad is not None
        warped = warp_to_rect(gray, quad)
        assert warped.shape[0] > 10 and warped.shape[1] > 10


class TestRectify:
    def test_recovers_dimensions_from_a_clean_grid(self):
        result = rectify(synthetic_grid(rows=11, cols=9))
        assert result is not None
        assert (result.rows, result.cols) == (11, 9)

    def test_recovers_dimensions_despite_keystone(self):
        # The whole point: a tilted grid still measures correctly, because it
        # is straightened before the lattice is measured.
        result = rectify(synthetic_grid(rows=11, cols=9, keystone=0.06))
        assert result is not None
        assert (result.rows, result.cols) == (11, 9)

    def test_recovers_dimensions_despite_low_contrast(self):
        # Faint grid lines, as in a washed-out photo of a phone screen.
        result = rectify(synthetic_grid(rows=11, cols=9, contrast=40))
        assert result is not None
        assert (result.rows, result.cols) == (11, 9)

    def test_returns_none_for_an_image_with_no_grid(self):
        assert rectify(np.full((300, 300, 3), 255, np.uint8)) is None

    def test_returns_none_for_an_empty_image(self):
        assert rectify(np.zeros((0, 0, 3), np.uint8)) is None
