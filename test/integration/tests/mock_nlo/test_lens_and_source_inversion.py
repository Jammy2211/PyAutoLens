from test.integration.tests import runner
from test.integration.tests.lens_and_source_inversion.adaptive_brightness import (
    lens_mass_x1_source_x1_hyper_constant_pass,
)


class TestCase:
    def _test_lens_mass_x1_source_x1_adaptive_weighted_hyper_constant_pass_mn(self):
        runner.run_with_multi_nest(lens_mass_x1_source_x1_hyper_constant_pass)

    def _test_lens_mass_x1_source_x1_adaptive_weighted_hyper_constant_pass_mock(self):
        runner.run_a_mock(lens_mass_x1_source_x1_hyper_constant_pass)
