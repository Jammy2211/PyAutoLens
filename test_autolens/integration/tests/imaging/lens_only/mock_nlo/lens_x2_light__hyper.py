from test_autolens.integration.tests.imaging.lens_only import lens_x2_light__hyper
from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test__lens_x2_light__hyper(self):
        run_a_mock(lens_x2_light__hyper)
