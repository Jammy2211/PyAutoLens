from test_autolens.integration.tests.imaging.lens__source_inversion.adaptive_brightness import (
    lens_mass__source__hyper_instance_pass,
)
from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test__lens_mass__source__hyper_instance_pass(self):
        run_a_mock(lens_mass__source__hyper_instance_pass)
