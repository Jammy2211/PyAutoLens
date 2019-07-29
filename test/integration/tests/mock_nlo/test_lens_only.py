from test.integration.tests.lens_only import lens_x1_galaxy
from test.integration.tests.lens_only import lens_x1_galaxy_hyper
from test.integration.tests.lens_only import lens_x1_galaxy_link_param
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_x1_galaxy(self):
        run_a_mock(lens_x1_galaxy)

    def _test_lens_x1_galaxy_hyper(self):
        run_a_mock(lens_x1_galaxy_hyper)

    def _test_lens_x1_galaxy_link_param(self):
        run_a_mock(lens_x1_galaxy_link_param)
