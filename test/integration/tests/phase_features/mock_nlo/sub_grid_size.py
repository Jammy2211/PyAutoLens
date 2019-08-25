from test.integration.tests.phase_features import sub_grid_size
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_sub_grid_size(self):
        run_a_mock(sub_grid_size)
