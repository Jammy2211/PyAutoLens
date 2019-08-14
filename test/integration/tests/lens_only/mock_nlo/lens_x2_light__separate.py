from test.integration.tests.lens_only import lens_x2_light__separate
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_x2_light__separate(self):
        run_a_mock(lens_x2_light__separate)
