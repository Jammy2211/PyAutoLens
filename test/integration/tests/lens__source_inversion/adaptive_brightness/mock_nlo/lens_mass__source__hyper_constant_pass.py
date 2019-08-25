from test.integration.tests.lens__source_inversion.adaptive_brightness import (
    lens_mass__source__hyper_constant_pass,
)
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_mass__source__hyper_constant_pass(self):
        run_a_mock(lens_mass__source__hyper_constant_pass)
