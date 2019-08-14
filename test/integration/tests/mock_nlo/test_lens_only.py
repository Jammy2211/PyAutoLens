from test.integration.tests.lens_only import lens_light
from test.integration.tests.lens_only import lens_light__hyper
from test.integration.tests.lens_only import lens_light__link_param
from test.integration.tests.runner import run_a_mock


class TestCase:
    def _test__lens_(self):
        run_a_mock(lens_light)

    def _test__lens__hyper(self):
        run_a_mock(lens_light__hyper)

    def _test__lens__link_param(self):
        run_a_mock(lens_light__link_param)
