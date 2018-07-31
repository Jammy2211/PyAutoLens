from src.imaging import image as im
from src.imaging import mask as msk
from src.imaging import convolution
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


@pytest.fixture(name="sparse_mask")
def make_sparse_mask(mask):
    return msk.SparseMask(mask, 1)


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

    def test_blurring_mask(self, masked_image):
        assert masked_image.blurring_mask.all()

    def test_convolvers(self, masked_image):
        assert type(masked_image.convolver_image) == convolution.ConvolverImage
        assert type(masked_image.convolver_mapping_matrix) == convolution.ConvolverMappingMatrix

    def test_map_to_2d(self, masked_image):
        assert (masked_image.map_to_2d(np.array([1, 1, 1, 1, 1])) == np.array([[0, 1, 0],
                                                                               [1, 1, 1],
                                                                               [0, 1, 0]])).all()

    def test_sparse_mask(self, mask, sparse_mask):
        assert (mask == sparse_mask).all()

    def test_subtract(self, masked_image):
        subtracted_image = masked_image - np.array([1, 0, 1, 0, 0])
        assert isinstance(subtracted_image, mi.MaskedImage)
        assert subtracted_image.psf == masked_image.psf
        assert subtracted_image.pixel_scale == masked_image.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1, 1])
