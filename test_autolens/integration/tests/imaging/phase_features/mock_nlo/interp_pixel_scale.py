from test import pixel_scales_interp

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_pixel_scales_interp(self):
        run_a_mock(pixel_scales_interp)
