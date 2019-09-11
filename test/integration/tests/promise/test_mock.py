import autolens as al
from test.integration.tests.promise import lens_mass__source_x2
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_mass__source_x2(self):
        run_a_mock(lens_mass__source_x2)
