from test import sub_size

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_sub_size(self):
        run_a_mock(sub_size)
