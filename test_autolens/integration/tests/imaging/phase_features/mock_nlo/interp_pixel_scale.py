from test import interpolation_pixel_scale

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_interpolation_pixel_scale(self):
        run_a_mock(interpolation_pixel_scale)
