from test import image_psf_shape

from test_autolens.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_image_psf_shape(self):
        run_a_mock(image_psf_shape)
