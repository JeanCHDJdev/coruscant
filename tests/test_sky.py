"""Unit tests for lightweight sky-mask helpers."""

import unittest

import numpy as np

try:
    import healpy  # noqa: F401
except ImportError:
    HEALPY_AVAILABLE = False
else:
    HEALPY_AVAILABLE = True

if HEALPY_AVAILABLE:
    from coruscant.sky.sky import get_fsky, sky_area, sky_overlap


@unittest.skipUnless(HEALPY_AVAILABLE, "healpy not installed")
class SkyMaskTests(unittest.TestCase):
    """Cover the simple mask helpers with standard-library unittest."""

    def test_get_fsky_counts_boolean_pixels(self) -> None:
        """Boolean masks should map directly to the visible-sky fraction."""
        mask = np.array([True, False, True, False])

        self.assertEqual(get_fsky(mask), 0.5)

    def test_sky_area_scales_full_sky_fraction(self) -> None:
        """The sky area should be the full-sky area scaled by the mask fraction."""
        mask = np.array([1.0, 1.0, 0.0, 0.0])
        expected_area = 0.5 * 4.0 * np.pi * (180.0 / np.pi) ** 2

        self.assertTrue(np.isclose(sky_area(mask), expected_area))

    def test_sky_overlap_marks_shared_pixels(self) -> None:
        """Overlap should keep only pixels visible in both masks."""
        mask1 = np.array([True, False, True, False])
        mask2 = np.array([True, True, False, False])

        np.testing.assert_array_equal(
            sky_overlap(mask1, mask2),
            np.array([True, False, False, False]),
        )