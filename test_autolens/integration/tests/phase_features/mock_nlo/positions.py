from test import positions
from test_autolens.integration.tests.runner import run_a_mock


class TestCase:
    def _test_positions(self):
        run_a_mock(positions)
