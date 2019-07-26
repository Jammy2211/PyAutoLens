from test.integration.tests.full_pipeline import hyper_with_lens_light_bg
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_hyper_with_lens_light(self):
        run_a_mock(hyper_with_lens_light_bg)
