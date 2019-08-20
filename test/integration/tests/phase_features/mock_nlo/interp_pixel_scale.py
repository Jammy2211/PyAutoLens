from test.integration.tests.phase_features import pixel_scale_interpolation_grid
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_pixel_scale_interpolation_grid(self):
        run_a_mock(pixel_scale_interpolation_grid)
