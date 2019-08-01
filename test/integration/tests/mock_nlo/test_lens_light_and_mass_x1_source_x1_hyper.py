from test.integration.tests.lens_and_source import (
    lens_light_and_mass_x1_source_x1_hyper,
)
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_mass_x1_source_x1_adaptive_weighted_hyper_constant_pass(self):
        run_a_mock(lens_light_and_mass_x1_source_x1_hyper)
