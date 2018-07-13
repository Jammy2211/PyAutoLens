from src.imaging import image as im
from src.imaging import mask as msk
from src.imaging import masked_image as mi
import numpy as np
import pytest


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(np.ones((1, 1)), 1)
    return im.Image(np.ones((3, 3)), effective_exposure_time=1., pixel_scale=1., psf=psf,
                    background_noise=np.ones((3, 3)), poisson_noise=np.ones((3, 3)))


@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, False, True],
                              [False, False, False],
                              [True, False, True]]))


@pytest.fixture(name="masked_image")
def make_masked_image(image, mask):
    return mi.MaskedImage(image, mask)


class TestMaskedImage(object):
    def test_attributes(self, image, masked_image):
        assert image.effective_exposure_time == masked_image.effective_exposure_time
        assert image.pixel_scale == masked_image.pixel_scale
        assert image.psf == masked_image.psf

    def test_masking(self, masked_image):
        assert masked_image.background_noise.shape == (5,)
        assert masked_image.poisson_noise.shape == (5,)

    def test_coordinate_grid(self, masked_image):
        assert masked_image.coordinate_grid.shape == (5, 2)
        assert (masked_image.coordinate_grid == np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])).all()

    def test_blurring_coordinate_grid(self, masked_image):
        assert masked_image.blurring_coordinate_grid.shape == (0, 2)

    def test_indices(self, masked_image):
        assert masked_image.border_pixel_indices.shape == (5,)

    def test_blurring_mask(self, masked_image):
        assert masked_image.blurring_mask.all()

    def test_kernel_convolver(self, masked_image):
        assert masked_image.kernel_convolver.length == 1

    def test_sub_coordinate_grid(self, masked_image):
        assert masked_image.sub_coordinate_grid.shape == (5, 2)
        assert (masked_image.sub_coordinate_grid == np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])).all()

    def test_sub_to_pixel(self, masked_image):
        assert (masked_image.sub_to_image == np.array(range(5))).all()

    def test_sub_data_to_image(self, masked_image):
        assert (masked_image.sub_data_to_image(np.array(range(5))) == np.array(range(5))).all()

    def test_map_to_2d(self, masked_image):
        assert (masked_image.map_to_2d(np.array([1, 1, 1, 1, 1])) == np.array([[0, 1, 0],
                                                                               [1, 1, 1],
                                                                               [0, 1, 0]])).all()
