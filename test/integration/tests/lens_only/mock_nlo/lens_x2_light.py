from test.integration.tests.lens_only import lens_x2_light
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_x2_light(self):
        run_a_mock(lens_x2_light)
