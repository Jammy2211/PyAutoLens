from test.integration.tests.lens__source_inversion.adaptive_magnification import (
    lens_mass__source,
)
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_mass__source(self):
        run_a_mock(lens_mass__source)
