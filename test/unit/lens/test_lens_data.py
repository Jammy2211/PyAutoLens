import numpy as np
import pytest

from autolens.data.array import scaled_array
from autolens.data import ccd, convolution
from autolens.data.array.util import grid_util
from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.lens import lens_data as ld
from autolens.model.inversion import convolution as inversion_convolution

from test.unit.mock.data import mock_grids

from test.unit.fixtures.data.fix_ccd import ccd_data
from test.unit.fixtures.data.fix_mask import mask

@pytest.fixture(name="lens_data")
def make_lens_ccd(ccd_data, mask):
    return ld.LensData(ccd_data=ccd_data, mask=mask)

class TestLensData(object):

    def test__attributes(self, ccd_data, lens_data):

        assert lens_data.pixel_scale == ccd_data.pixel_scale
        assert lens_data.pixel_scale == 1.0

        assert (lens_data.unmasked_image == ccd_data.image).all()
        assert (lens_data.unmasked_image == np.ones((5,5))).all()

        assert (lens_data.unmasked_noise_map == ccd_data.noise_map).all()
        assert (lens_data.unmasked_noise_map == 2.0*np.ones((5,5))).all()

        assert (lens_data.psf == ccd_data.psf).all()
        assert (lens_data.psf == np.ones((3,3))).all()

        assert lens_data.image_psf_shape == (3,3)
        assert lens_data.inversion_psf_shape == (3,3)

    def test__masking(self, lens_data):

        assert (lens_data.image_1d == np.ones(9)).all()
        assert (lens_data.noise_map_1d == 2.0*np.ones(9)).all()
        assert (lens_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (lens_data.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        assert (lens_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        assert (lens_data.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

    def test__grid_stack(self, lens_data):

        assert (lens_data.grid_stack.regular == mock_grids.MockRegularGrid()).all()
        assert (lens_data.grid_stack.sub == mock_grids.MockSubGrid()).all()
        assert (lens_data.grid_stack.blurring == mock_grids.MockBlurringGrid()).all()

    def test__padded_grid_stack(self, lens_data):

        padded_image_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((7, 7), False), pixel_scales=lens_data.unmasked_image.pixel_scales)

        assert (lens_data.padded_grid_stack.regular == padded_image_util).all()
        assert lens_data.padded_grid_stack.regular.image_shape == (5, 5)
        assert lens_data.padded_grid_stack.regular.padded_shape == (7, 7)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((7, 7), False), pixel_scales=lens_data.unmasked_image.pixel_scales,
            sub_grid_size=lens_data.grid_stack.sub.sub_grid_size)

        assert lens_data.padded_grid_stack.sub == pytest.approx(padded_sub_util, 1e-4)
        assert lens_data.padded_grid_stack.sub.image_shape == (5, 5)
        assert lens_data.padded_grid_stack.sub.padded_shape == (7, 7)

        assert (lens_data.padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()

    def test__interp_pixel_scale_input__grid_stack_and_padded_grid_stack_include_interpolators(self, ccd_data, mask):

        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, interp_pixel_scale=1.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask, sub_grid_size=2, psf_shape=(3, 3))
        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (lens_data.grid_stack.regular == new_grid_stack.regular).all()
        assert (lens_data.grid_stack.regular.interpolator.vtx == new_grid_stack.regular.interpolator.vtx).all()
        assert (lens_data.grid_stack.regular.interpolator.wts == new_grid_stack.regular.interpolator.wts).all()

        assert (lens_data.grid_stack.sub == new_grid_stack.sub).all()
        assert (lens_data.grid_stack.sub.interpolator.vtx == new_grid_stack.sub.interpolator.vtx).all()
        assert (lens_data.grid_stack.sub.interpolator.wts == new_grid_stack.sub.interpolator.wts).all()
        
        assert (lens_data.grid_stack.blurring == new_grid_stack.blurring).all()
        assert (lens_data.grid_stack.blurring.interpolator.vtx == new_grid_stack.blurring.interpolator.vtx).all()
        assert (lens_data.grid_stack.blurring.interpolator.wts == new_grid_stack.blurring.interpolator.wts).all()

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask, sub_grid_size=2, psf_shape=(3, 3))
        new_padded_grid_stack = \
            padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (lens_data.padded_grid_stack.regular == new_padded_grid_stack.regular).all()
        assert (lens_data.padded_grid_stack.regular.interpolator.vtx == new_padded_grid_stack.regular.interpolator.vtx).all()
        assert (lens_data.padded_grid_stack.regular.interpolator.wts == new_padded_grid_stack.regular.interpolator.wts).all()

        assert (lens_data.padded_grid_stack.sub == new_padded_grid_stack.sub).all()
        assert (lens_data.padded_grid_stack.sub.interpolator.vtx == new_padded_grid_stack.sub.interpolator.vtx).all()
        assert (lens_data.padded_grid_stack.sub.interpolator.wts == new_padded_grid_stack.sub.interpolator.wts).all()

        assert (lens_data.padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()

    def test__border(self, lens_data):

        assert (lens_data.border == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__convolvers(self, lens_data):
        assert type(lens_data.convolver_image) == convolution.ConvolverImage
        assert type(lens_data.convolver_mapping_matrix) == inversion_convolution.ConvolverMappingMatrix

    def test__uses_inversion__does_not_create_mapping_matrix_conovolver_if_false(self, ccd_data, mask):

        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, uses_inversion=False)

        assert lens_data.convolver_mapping_matrix == None

    def test__different_ccd_data_without_mock_objects__customize_constructor_inputs(self):

        psf = ccd.PSF(np.ones((7, 7)), 1)
        ccd_data = ccd.CCDData(np.ones((19, 19)), pixel_scale=3., psf=psf, noise_map=2.0*np.ones((19, 19)))
        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(19, 19), pixel_scale=1.0, invert=True)
        mask[9, 9] = False

        lens_data = ld.LensData(ccd_data, mask, sub_grid_size=8, image_psf_shape=(5, 5),
                                inversion_psf_shape=(3, 3), positions=[np.array([[1.0, 1.0]])])

        assert (lens_data.unmasked_image == np.ones((19, 19))).all()
        assert (lens_data.unmasked_noise_map == 2.0*np.ones((19, 19))).all()
        assert (lens_data.psf == np.ones((7,7))).all()

        assert lens_data.sub_grid_size == 8
        assert lens_data.convolver_image.psf_shape == (5, 5)
        assert lens_data.convolver_mapping_matrix.psf_shape == (3, 3)
        assert (lens_data.positions[0] == np.array([[1.0, 1.0]])).all()

        assert lens_data.image_psf_shape == (5,5)
        assert lens_data.inversion_psf_shape == (3, 3)

    def test__lens_data_with_modified_image(self, lens_data):

        lens_data = lens_data.new_lens_data_with_modified_image(modified_image=8.0 * np.ones((5, 5)))

        assert (lens_data.unmasked_image == 8.0*np.ones((5,5))).all()

        assert (lens_data.image_1d == 8.0*np.ones(9)).all()

        assert (lens_data.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__lens_data_with_binned_up_ccd_data(self, ccd_data):

        # Easier to test using a 6x6 image, noise-map, etc.

        ccd_data.image = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)
        ccd_data.noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)
        ccd_data.background_noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)
        ccd_data.poisson_noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)
        ccd_data.exposure_time_map = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)
        ccd_data.background_sky_map = scaled_array.ScaledSquarePixelArray(array=np.ones((6,6)), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True ,True, True, True],
                                        [True, True, True ,True, True, True],
                                        [True, True, False, False, True, True],
                                        [True, True, False, False, True, True],
                                        [True, True, True ,True, True, True],
                                        [True, True, True ,True, True, True]]), pixel_scale=1.0)

        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

        binned_up_psf = lens_data.ccd_data.psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5)

        lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(bin_up_factor=2)

        assert (lens_data.unmasked_image == np.ones((3,3))).all()
        assert (lens_data.psf == binned_up_psf).all()
        assert (lens_data.unmasked_noise_map == 0.5*np.ones((3,3))).all()
        assert (lens_data.ccd_data.background_noise_map == 0.5*np.ones((3,3))).all()
        assert (lens_data.ccd_data.poisson_noise_map == 0.5*np.ones((3,3))).all()
        assert (lens_data.ccd_data.exposure_time_map == 4.0*np.ones((3,3))).all()
        assert (lens_data.ccd_data.background_sky_map == np.ones((3,3))).all()

        assert (lens_data.mask_2d == np.array([[True, True, True],
                                               [True, False, True],
                                               [True, True, True]])).all()

        assert (lens_data.image_1d == np.ones((1))).all()
        assert (lens_data.noise_map_1d == 0.5*np.ones((1))).all()