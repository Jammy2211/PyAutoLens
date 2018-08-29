from autolens.imaging import imaging_util
from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.imaging import convolution
from autolens.imaging import masked_image as mi
import numpy as np
import pytest


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(array=np.ones((3, 3)), renormalize=False)
    return im.Image(np.ones((4, 4)), pixel_scale=3., psf=psf, noise=np.ones((4, 4)))


@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask( np.array([[True,  True,  True, True],
                               [True, False, False, True],
                               [True, False, False, True],
                               [True,  True,  True, True]]), pixel_scale=3.0)


@pytest.fixture(name="masked_image")
def make_masked_image(image, mask):
    return mi.MaskedImage(image=image, mask=mask)


class TestMaskedImage(object):

    def test_attributes(self, image, masked_image):
        assert image.pixel_scale == masked_image.pixel_scale
        assert (image.psf == masked_image.psf).all()

    def test_masking(self, masked_image):
        assert masked_image.noise.shape == (4,)

    def test_grids(self, masked_image):

        assert masked_image.grids.image.shape == (4, 2)

        assert (masked_image.grids.image == np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])).all()
        assert (masked_image.grids.sub == np.array([[-2.0, -2.0], [-2.0, -1.0], [-1.0, -2.0], [-1.0, -1.0],
                                                    [-2.0, 1.0], [-2.0, 2.0], [-1.0, 1.0], [-1.0, 2.0],
                                                    [1.0, -2.0], [1.0, -1.0], [2.0, -2.0], [2.0, -1.0],
                                                    [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])).all()
        assert (masked_image.grids.blurring == np.array([[-4.5, -4.5], [-4.5, -1.5], [-4.5, 1.5], [-4.5, 4.5],
                                                         [-1.5, -4.5],  [-1.5, 4.5],  [1.5, -4.5], [1.5, 4.5],
                                                          [4.5, -4.5],  [4.5, -1.5],  [4.5, 1.5],  [4.5, 4.5]])).all()

    def test_mapper_grids(self, masked_image):

        image_mapper_util = imaging_util.image_grid_masked_from_mask_and_pixel_scale(np.full((6, 6), False),
                                                                                     masked_image.image.pixel_scale)

        sub_mapper_util = imaging_util.sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(np.full((6, 6), False),
                                                 masked_image.image.pixel_scale, masked_image.grids.sub.sub_grid_size)

        assert (masked_image.grid_mappers.image == image_mapper_util).all()
        assert masked_image.grid_mappers.image.original_shape == (4,4)
        assert masked_image.grid_mappers.image.padded_shape == (6,6)

        assert (masked_image.grid_mappers.sub == sub_mapper_util).all()
        assert masked_image.grid_mappers.sub.original_shape == (4,4)
        assert masked_image.grid_mappers.sub.padded_shape == (6,6)

        assert (masked_image.grid_mappers.blurring == np.array([[0.0, 0.0]])).all()

    def test_borders(self, masked_image):

        assert (masked_image.borders.image == np.array([0, 1, 2, 3])).all()
        assert (masked_image.borders.sub == np.array([0, 5, 10, 15])).all()

    def test_blurring_mask(self, masked_image):
        assert (masked_image.blurring_mask == np.array([[False, False, False, False],
                                                        [False,  True,  True, False],
                                                        [False,  True,  True, False],
                                                        [False, False, False, False]])).all()

    def test_convolvers(self, masked_image):
        assert type(masked_image.convolver_image) == convolution.ConvolverImage
        assert type(masked_image.convolver_mapping_matrix) == convolution.ConvolverMappingMatrix

    def test_subtract(self, masked_image):
        subtracted_image = masked_image - np.array([1, 0, 1, 0])
        assert isinstance(subtracted_image, mi.MaskedImage)
        assert (subtracted_image.psf == masked_image.psf).all()
        assert subtracted_image.pixel_scale == masked_image.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1])

    def test__constructor_inputs(self):

        psf = im.PSF(np.ones((7, 7)), 1)
        image = im.Image(np.ones((51, 51)), pixel_scale=3., psf=psf, noise=np.ones((51, 51)))
        mask = msk.Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds=(51.0, 51.0), pixel_scale=1.0)
        mask[26, 26] = False

        masked_image = mi.MaskedImage(image, mask, sub_grid_size=8, image_psf_shape=(5, 5),
                                      pixelization_psf_shape=(3, 3), positions=[np.array([[1.0, 1.0]])])

        assert masked_image.sub_grid_size == 8
        assert masked_image.convolver_image.psf_shape == (5,5)
        assert masked_image.convolver_mapping_matrix.psf_shape == (3, 3)
        assert (masked_image.positions[0] == np.array([[1.0, 1.0]])).all()