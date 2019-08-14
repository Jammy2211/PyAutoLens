from test.integration.tests.lens_and_source import (
    lens_light_mass__source_hyper,
)
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_mass__source_adaptive_weighted_hyper_constant_pass(self):
        run_a_mock(lens_light_mass__source_hyper)
