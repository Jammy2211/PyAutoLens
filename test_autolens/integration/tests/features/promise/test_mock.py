from test_autolens.integration.tests.imaging.lens__source import lens_mass__source_x2
from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_lens_mass__source_x2(self):
        run_a_mock(lens_mass__source_x2)
