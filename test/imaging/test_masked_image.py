from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.imaging import convolution
from autolens.imaging import masked_image as mi
import numpy as np
import pytest


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(np.ones((1, 1)), 1)
    return im.Image(np.ones((3, 3)), pixel_scale=3., psf=psf, noise=np.ones((3, 3)))


@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, False, True],
                              [False, False, False],
                              [True, False, True]]), pixel_scale=3.0)


@pytest.fixture(name="sparse_mask")
def make_sparse_mask(mask):
    return msk.SparseMask(mask, 1.0)


@pytest.fixture(name="masked_image")
def make_masked_image(image, mask):
    return mi.MaskedImage(image, mask)


class TestMaskedImage(object):
    def test_attributes(self, image, masked_image):
        assert image.pixel_scale == masked_image.pixel_scale
        assert image.psf == masked_image.psf

    def test_masking(self, masked_image):
        assert masked_image.noise.shape == (5,)

    def test_grids(self, masked_image):

        assert masked_image.grids.image.shape == (5, 2)

        assert (masked_image.grids.image == np.array([[-3, 0], [0, -3], [0, 0], [0, 3], [3, 0]])).all()
        assert (masked_image.grids.sub == np.array([[-3.5, -0.5], [-3.5, 0.5], [-2.5, -0.5], [-2.5, 0.5],
                                                    [-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5],
                                                    [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                                                    [-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5],
                                                    [2.5, -0.5], [2.5, 0.5], [3.5, -0.5], [3.5, 0.5]])).all()

    def test_borders(self, masked_image):

        assert (masked_image.borders.image == np.array([0, 1, 2, 3, 4])).all()
        assert (masked_image.borders.sub == np.array([0, 4, 8, 13, 18])).all()

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
