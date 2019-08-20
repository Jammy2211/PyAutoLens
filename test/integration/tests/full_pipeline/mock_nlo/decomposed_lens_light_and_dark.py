from test.integration.tests.full_pipeline import decomposed_lens_light_and_dark
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_decomposed_lens_light_and_dark(self):
        run_a_mock(decomposed_lens_light_and_dark)
