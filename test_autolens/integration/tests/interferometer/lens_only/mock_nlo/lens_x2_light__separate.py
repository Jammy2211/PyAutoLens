from test_autolens.integration.tests.interferometer.lens_only import (
    lens_x2_light__separate,
)
from test_autolens.integration.tests.interferometer.runner import run_a_mock


class TestCase:
    def _test__lens_x2_light__separate(self):
        run_a_mock(lens_x2_light__separate)
