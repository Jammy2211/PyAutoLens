from test.integration.tests.phase_features import positions
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_positions(self):
        run_a_mock(positions)
