from test.integration.tests.lens__source import lens_light_mass__source
from test.integration.tests.lens__source import lens_light_mass__source__hyper
from test.integration.tests.lens__source import lens_light_mass__source__hyper_bg
from test.integration.tests.lens__source import lens_mass__source_x2
from test.integration.tests.lens__source import lens_mass__source_x2__hyper
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_light_mass__source(self):
        run_a_mock(lens_light_mass__source)

    def _test_lens_light_mass__source__hyper(self):
        run_a_mock(lens_light_mass__source__hyper)

    def _test_lens_light_mass__source__hyper_bg(self):
        run_a_mock(lens_light_mass__source__hyper_bg)

    def _test_lens_mass__source_x2(self):
        run_a_mock(lens_mass__source_x2)

    def _test_lens_mass__source_x2__hyper(self):
        run_a_mock(lens_mass__source_x2__hyper)
