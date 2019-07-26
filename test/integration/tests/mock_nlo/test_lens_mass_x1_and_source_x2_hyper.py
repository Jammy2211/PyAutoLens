from test.integration.tests.lens_and_source import lens_mass_x1_source_x2_hyper
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_mass_x1_source_x2_hyper(self):
        run_a_mock(lens_mass_x1_source_x2_hyper)
