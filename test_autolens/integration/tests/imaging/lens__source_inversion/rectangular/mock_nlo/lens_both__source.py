from test_autolens.integration.tests.imaging.lens__source_inversion.rectangular import (
    lens_both__source,
)
from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test__lens_both__source(self):
        run_a_mock(lens_both__source)
