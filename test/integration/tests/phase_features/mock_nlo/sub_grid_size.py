from test.integration.tests.phase_features import sub_size
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_sub_size(self):
        run_a_mock(sub_size)
