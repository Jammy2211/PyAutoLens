from test_autolens.integration.tests.interferometer.lens__source_inversion.adaptive_brightness import (
    lens_mass__source__hyper_bg,
)
from test_autolens.integration.tests.interferometer.runner import run_a_mock


class TestCase:
    def _test__lens_mass__source__hyper_bg(self):
        run_a_mock(lens_mass__source__hyper_bg)
