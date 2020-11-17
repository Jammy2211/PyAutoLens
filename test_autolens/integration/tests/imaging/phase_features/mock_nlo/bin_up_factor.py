from test import bin_up_factor

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_bin_up_factor(self):
        run_a_mock(bin_up_factor)
