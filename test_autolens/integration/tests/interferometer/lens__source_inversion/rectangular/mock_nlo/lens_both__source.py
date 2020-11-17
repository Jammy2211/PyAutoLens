from test_autolens.integration.tests.interferometer.lens__source_inversion.rectangular import (
    lens_both__source,
)
from test_autolens.integration.tests.interferometer.runner import run_a_mock


class TestCase:
    def _test__lens_both__source(self):
        run_a_mock(lens_both__source)
