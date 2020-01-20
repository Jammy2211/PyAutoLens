from test_autolens.integration.tests.imaging.full_pipeline import hyper_no_lens_light_bg
from test_autolens.integration.tests.imaging.full_pipeline import (
    hyper_no_lens_light_bg_extension_false,
)
from test_autolens.integration.tests.imaging.full_pipeline import (
    hyper_with_lens_light_bg,
)
from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_hyper_no_lens_light_bg_extension_false(self):
        run_a_mock(hyper_no_lens_light_bg_extension_false)

    def _test_hyper_no_lens_light_bg(self):
        run_a_mock(hyper_no_lens_light_bg)

    def _test_hyper_with_lens_light(self):
        run_a_mock(hyper_with_lens_light_bg)

    def _test_hyper_with_lens_light_bg(self):
        run_a_mock(hyper_with_lens_light_bg)

    def _test_hyper_with_lens_light_bg_new_api(self):
        run_a_mock(hyper_with_lens_light_bg)
