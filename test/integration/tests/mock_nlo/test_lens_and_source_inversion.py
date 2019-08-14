from test.integration.tests import runner
from test.integration.tests.lens_and_source_inversion.adaptive_brightness import (
    lens_mass__source_hyper_constant_pass,
)


class TestCase:
    def _test__lens_mass__source_adaptive_weighted_hyper_constant_pass_mn(self):
        runner.run_with_multi_nest(lens_mass__source_hyper_constant_pass)

    def _test__lens_mass__source_adaptive_weighted_hyper_constant_pass_mock(self):
        runner.run_a_mock(lens_mass__source_hyper_constant_pass)
