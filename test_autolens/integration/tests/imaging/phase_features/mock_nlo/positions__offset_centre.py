from test import positions__offset_centre

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_positions__offset_centre(self):
        run_a_mock(positions__offset_centre)
