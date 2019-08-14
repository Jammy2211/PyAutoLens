from test.integration.tests.phase_features import galaxy_redshifts
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_galaxy_redshifts(self):
        run_a_mock(galaxy_redshifts)
