import numpy as np
import pytest

from autolens.imaging import scaled_array as sca
from autolens.imaging import imaging_util
from autolens.imaging import mask as msk
from autolens.galaxy import galaxy_data as gd

@pytest.fixture(name='scaled_array')
def make_scaled_array():
    return sca.ScaledSquarePixelArray(array=np.ones((4, 4)), pixel_scale=3.0)

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=3.0)

@pytest.fixture(name="galaxy_data")
def make_lensing_image(scaled_array, mask):
    return gd.GalaxyData(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask)


class TestGalaxyData(object):

    def test_attributes(self, scaled_array, galaxy_data):
        assert scaled_array.pixel_scale == galaxy_data.pixel_scale

    def test__scaled_array_and_mapper(self, galaxy_data):
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map == 2.0*np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

    def test_grids(self, galaxy_data):

        assert galaxy_data.grids.image.shape == (4, 2)

        assert (galaxy_data.grids.image == np.array([[1.5, -1.5], [1.5, 1.5],
                                                       [-1.5, -1.5], [-1.5, 1.5]])).all()
        assert (galaxy_data.grids.sub == np.array([[2.0, -2.0], [2.0, -1.0], [1.0, -2.0], [1.0, -1.0],
                                                     [2.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 2.0],
                                                     [-1.0, -2.0], [-1.0, -1.0], [-2.0, -2.0], [-2.0, -1.0],
                                                     [-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0], [-2.0, 2.0]])).all()

    def test_unmasked_grids(self, galaxy_data):

        padded_image_util = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=np.full((4, 4), False),
                          pixel_scales=galaxy_data.array.pixel_scales)

        assert (galaxy_data.unmasked_grids.image == padded_image_util).all()
        assert galaxy_data.unmasked_grids.image.image_shape == (4, 4)
        assert galaxy_data.unmasked_grids.image.padded_shape == (4, 4)

        padded_sub_util = imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False), pixel_scales=galaxy_data.array.pixel_scales,
            sub_grid_size=galaxy_data.grids.sub.sub_grid_size)

        assert galaxy_data.unmasked_grids.sub == pytest.approx(padded_sub_util, 1e-4)
        assert galaxy_data.unmasked_grids.sub.image_shape == (4, 4)
        assert galaxy_data.unmasked_grids.sub.padded_shape == (4, 4)

    def test_subtract(self, galaxy_data):
        subtracted_image = galaxy_data - np.array([1, 0, 1, 0])
        assert isinstance(subtracted_image, gd.GalaxyData)
        assert subtracted_image.pixel_scale == galaxy_data.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1])