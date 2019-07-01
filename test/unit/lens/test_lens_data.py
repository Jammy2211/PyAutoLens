import numpy as np
import pytest

from autolens.data import convolution
from autolens.data.array.util import grid_util
from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.lens import lens_data as ld
from autolens.model.inversion import convolution as inversion_convolution

@pytest.fixture(name="lens_data_5x5")
def make_lens_data_5x5(ccd_data_5x5, mask_5x5):
    return ld.LensData(ccd_data=ccd_data_5x5, mask=mask_5x5)

@pytest.fixture(name="lens_data_6x6")
def make_lens_data_6x6(ccd_data_6x6, mask_6x6):
    return ld.LensData(ccd_data=ccd_data_6x6, mask=mask_6x6)

class TestLensData(object):

    def test__attributes(
            self, ccd_data_5x5, lens_data_5x5):

        assert lens_data_5x5.pixel_scale == ccd_data_5x5.pixel_scale
        assert lens_data_5x5.pixel_scale == 1.0

        assert (lens_data_5x5.unmasked_image == ccd_data_5x5.image).all()
        assert (lens_data_5x5.unmasked_image == np.ones((5,5))).all()

        assert (lens_data_5x5.unmasked_noise_map == ccd_data_5x5.noise_map).all()
        assert (lens_data_5x5.unmasked_noise_map == 2.0*np.ones((5,5))).all()

        assert (lens_data_5x5.psf == ccd_data_5x5.psf).all()
        assert (lens_data_5x5.psf == np.ones((3,3))).all()

        assert lens_data_5x5.image_psf_shape == (3,3)
        assert lens_data_5x5.inversion_psf_shape == (3,3)

    def test__masking(
            self, lens_data_5x5):

        assert (lens_data_5x5.mask_1d == np.full(fill_value=False, shape=(9))).all()
        assert (lens_data_5x5.image_1d == np.ones(9)).all()
        assert (lens_data_5x5.noise_map_1d == 2.0*np.ones(9)).all()

        assert (lens_data_5x5.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

        assert (lens_data_5x5.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        assert (lens_data_5x5.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__grid_stack(
            self, lens_data_5x5, regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):

        assert (lens_data_5x5.grid_stack.regular == regular_grid_5x5).all()
        assert (lens_data_5x5.grid_stack.sub == sub_grid_5x5).all()
        assert (lens_data_5x5.grid_stack.blurring == blurring_grid_5x5).all()

    def test__padded_grid_stack(
            self, lens_data_5x5):

        padded_image_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((7, 7), False), pixel_scales=lens_data_5x5.unmasked_image.pixel_scales)

        assert (lens_data_5x5.padded_grid_stack.regular == padded_image_util).all()
        assert lens_data_5x5.padded_grid_stack.regular.image_shape == (5, 5)
        assert lens_data_5x5.padded_grid_stack.regular.padded_shape == (7, 7)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((7, 7), False), pixel_scales=lens_data_5x5.unmasked_image.pixel_scales,
            sub_grid_size=lens_data_5x5.grid_stack.sub.sub_grid_size)

        assert lens_data_5x5.padded_grid_stack.sub == pytest.approx(padded_sub_util, 1e-4)
        assert lens_data_5x5.padded_grid_stack.sub.image_shape == (5, 5)
        assert lens_data_5x5.padded_grid_stack.sub.padded_shape == (7, 7)

        assert (lens_data_5x5.padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()

    def test__interp_pixel_scale_input__grid_stack_and_padded_grid_stack_include_interpolators(
            self, ccd_data_5x5, mask_5x5):

        lens_data_5x5 = ld.LensData(ccd_data=ccd_data_5x5, mask=mask_5x5, interp_pixel_scale=1.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask_5x5, sub_grid_size=2, psf_shape=(3, 3))
        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (lens_data_5x5.grid_stack.regular == new_grid_stack.regular).all()
        assert (lens_data_5x5.grid_stack.regular.interpolator.vtx == new_grid_stack.regular.interpolator.vtx).all()
        assert (lens_data_5x5.grid_stack.regular.interpolator.wts == new_grid_stack.regular.interpolator.wts).all()

        assert (lens_data_5x5.grid_stack.sub == new_grid_stack.sub).all()
        assert (lens_data_5x5.grid_stack.sub.interpolator.vtx == new_grid_stack.sub.interpolator.vtx).all()
        assert (lens_data_5x5.grid_stack.sub.interpolator.wts == new_grid_stack.sub.interpolator.wts).all()
        
        assert (lens_data_5x5.grid_stack.blurring == new_grid_stack.blurring).all()
        assert (lens_data_5x5.grid_stack.blurring.interpolator.vtx == new_grid_stack.blurring.interpolator.vtx).all()
        assert (lens_data_5x5.grid_stack.blurring.interpolator.wts == new_grid_stack.blurring.interpolator.wts).all()

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask_5x5, sub_grid_size=2, psf_shape=(3, 3))
        new_padded_grid_stack = \
            padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (lens_data_5x5.padded_grid_stack.regular == new_padded_grid_stack.regular).all()
        assert (lens_data_5x5.padded_grid_stack.regular.interpolator.vtx == new_padded_grid_stack.regular.interpolator.vtx).all()
        assert (lens_data_5x5.padded_grid_stack.regular.interpolator.wts == new_padded_grid_stack.regular.interpolator.wts).all()

        assert (lens_data_5x5.padded_grid_stack.sub == new_padded_grid_stack.sub).all()
        assert (lens_data_5x5.padded_grid_stack.sub.interpolator.vtx == new_padded_grid_stack.sub.interpolator.vtx).all()
        assert (lens_data_5x5.padded_grid_stack.sub.interpolator.wts == new_padded_grid_stack.sub.interpolator.wts).all()

        assert (lens_data_5x5.padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()

    def test__cluster_pixel_scale_is_input__correct_cluster_bin_up_calculated__inversion_max_pixels_changes_bin_up(
            self, ccd_data_5x5, mask_5x5, regular_grid_5x5):

        ccd_data_5x5.pixel_scale = 1.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=1.0)

        assert lens_data_5x5.cluster.bin_up_factor == 1
        assert (lens_data_5x5.mask_2d == lens_data_5x5.cluster.mask).all()
        assert (lens_data_5x5.cluster == regular_grid_5x5).all()
        assert (lens_data_5x5.cluster.cluster_to_regular_all == np.array(
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]])).all()

        ccd_data_5x5.pixel_scale = 1.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=1.9)

        assert lens_data_5x5.cluster.bin_up_factor == 1
        assert (lens_data_5x5.mask_2d == lens_data_5x5.cluster.mask).all()
        assert (lens_data_5x5.cluster.cluster_to_regular_all == np.array(
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]])).all()

        ccd_data_5x5.pixel_scale = 1.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=2.0)
        assert lens_data_5x5.cluster.bin_up_factor == 2
        assert (lens_data_5x5.cluster.mask == np.array([[True, True, True],
                                                           [True, False, False],
                                                           [True, False, False]])).all()
        assert (lens_data_5x5.cluster == np.array(
            [[0.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [-2.0, 2.0]])).all()
        assert (lens_data_5x5.cluster.cluster_to_regular_all == np.array(
            [[0, 1, 3, 4],
            [2, 5, -1, -1],
             [6, 7, -1, -1],
             [8, -1, -1, -1]])).all()

        ccd_data_5x5.pixel_scale = 2.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=1.0)

        assert lens_data_5x5.cluster.bin_up_factor == 1

        ccd_data_5x5.pixel_scale = 1.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=None)

        assert lens_data_5x5.cluster.bin_up_factor == 1
        assert (lens_data_5x5.mask_2d == lens_data_5x5.cluster.mask).all()
        assert (lens_data_5x5.cluster == regular_grid_5x5).all()
        assert (lens_data_5x5.cluster.cluster_to_regular_all == np.array(
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]])).all()

        ccd_data_5x5.pixel_scale = 1.0
        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_scale=2.0, cluster_pixel_limit=5)

        assert lens_data_5x5.cluster.bin_up_factor == 1
        assert (lens_data_5x5.mask_2d == lens_data_5x5.cluster.mask).all()
        assert (lens_data_5x5.cluster.cluster_to_regular_all == np.array(
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]])).all()


    def test__border(
            self, lens_data_5x5):

        assert (lens_data_5x5.border == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__convolvers(
            self, lens_data_5x5):
        assert type(lens_data_5x5.convolver_image) == convolution.ConvolverImage
        assert type(lens_data_5x5.convolver_mapping_matrix) == inversion_convolution.ConvolverMappingMatrix

    def test__inversion_max_pixels(
            self, ccd_data_5x5, mask_5x5):

        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_limit=2)

        assert lens_data_5x5.cluster_pixel_limit == 2

        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5, cluster_pixel_limit=5)

        assert lens_data_5x5.cluster_pixel_limit == 5

    def test__uses_inversion__does_not_create_mapping_matrix_conovolver_if_false(
            self, ccd_data_5x5, mask_5x5):

        lens_data_5x5 = ld.LensData(ccd_data=ccd_data_5x5, mask=mask_5x5, uses_inversion=False)

        assert lens_data_5x5.convolver_mapping_matrix == None

    def test__uses_cluster_inversion__does_not_create_cluster_grid_if_false(
            self, ccd_data_5x5, mask_5x5):

        lens_data_5x5 = ld.LensData(
            ccd_data=ccd_data_5x5, mask=mask_5x5,
            cluster_pixel_scale=1.0, cluster_pixel_limit=1, uses_cluster_inversion=False)

        assert lens_data_5x5.cluster == None
        assert lens_data_5x5.cluster_pixel_scale == None
        assert lens_data_5x5.cluster_pixel_limit == None

    def test__different_ccd_data_without_mock_objects__customize_constructor_inputs(self):

        psf = ccd.PSF(np.ones((7, 7)), 1)
        ccd_data = ccd.CCDData(np.ones((19, 19)), pixel_scale=3., psf=psf, noise_map=2.0*np.ones((19, 19)))
        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(19, 19), pixel_scale=1.0, invert=True)
        mask[9, 9] = False

        lens_data_5x5 = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=8, image_psf_shape=(5, 5),
                                inversion_psf_shape=(3, 3), positions=[np.array([[1.0, 1.0]])])

        assert (lens_data_5x5.unmasked_image == np.ones((19, 19))).all()
        assert (lens_data_5x5.unmasked_noise_map == 2.0*np.ones((19, 19))).all()
        assert (lens_data_5x5.psf == np.ones((7,7))).all()

        assert lens_data_5x5.sub_grid_size == 8
        assert lens_data_5x5.convolver_image.psf_shape == (5, 5)
        assert lens_data_5x5.convolver_mapping_matrix.psf_shape == (3, 3)
        assert (lens_data_5x5.positions[0] == np.array([[1.0, 1.0]])).all()

        assert lens_data_5x5.image_psf_shape == (5,5)
        assert lens_data_5x5.inversion_psf_shape == (3, 3)

    def test__lens_data_5x5_with_modified_image(
            self, lens_data_5x5):

        lens_data_5x5 = lens_data_5x5.new_lens_data_with_modified_image(modified_image=8.0 * np.ones((5, 5)))

        assert (lens_data_5x5.unmasked_image == 8.0*np.ones((5,5))).all()

        assert (lens_data_5x5.image_1d == 8.0*np.ones(9)).all()

        assert (lens_data_5x5.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 8.0, 8.0, 8.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__lens_data_6x6_with_binned_up_ccd_data(
            self, lens_data_6x6):

        binned_up_psf = lens_data_6x6.ccd_data.psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5)

        lens_data_6x6 = lens_data_6x6.new_lens_data_with_binned_up_ccd_data_and_mask(bin_up_factor=2)

        assert (lens_data_6x6.unmasked_image == np.ones((3,3))).all()
        assert (lens_data_6x6.psf == binned_up_psf).all()
        assert (lens_data_6x6.unmasked_noise_map == np.ones((3,3))).all()
        assert (lens_data_6x6.ccd_data.background_noise_map == 1.5*np.ones((3,3))).all()
        assert (lens_data_6x6.ccd_data.poisson_noise_map == 2.0*np.ones((3,3))).all()
        assert (lens_data_6x6.ccd_data.exposure_time_map == 20.0*np.ones((3,3))).all()
        assert (lens_data_6x6.ccd_data.background_sky_map == 6.0*np.ones((3,3))).all()

        assert (lens_data_6x6.mask_2d == np.array([[True, True, True],
                                               [True, False, True],
                                               [True, True, True]])).all()

        assert (lens_data_6x6.image_1d == np.ones((1))).all()
        assert (lens_data_6x6.noise_map_1d == np.ones((1))).all()