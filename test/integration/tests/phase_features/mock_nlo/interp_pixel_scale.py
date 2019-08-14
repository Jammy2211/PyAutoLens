from test.integration.tests.phase_features import interp_pixel_scale
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_interp_pixel_scale(self):
        run_a_mock(interp_pixel_scale)
