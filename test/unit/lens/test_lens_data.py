import numpy as np
import pytest

from autolens.data import ccd, convolution
from autolens.data.array.util import grid_util
from autolens.data.array import grids
from autolens.data.array import scaled_array
from autolens.data.array import mask as msk
from autolens.lens import lens_data as ld
from autolens.model.inversion import convolution as inversion_convolution


@pytest.fixture(name='ccd')
def make_ccd():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((6, 6)), pixel_scale=3.0)
    psf = ccd.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
    noise_map = ccd.NoiseMap(array=2.0 * np.ones((6, 6)), pixel_scale=3.0)
    background_noise_map = ccd.NoiseMap(array=3.0 * np.ones((6, 6)), pixel_scale=3.0)
    poisson_noise_map = ccd.PoissonNoiseMap(array=4.0 * np.ones((6, 6)), pixel_scale=3.0)
    exposure_time_map = ccd.ExposureTimeMap(array=5.0 * np.ones((6, 6)), pixel_scale=3.0)
    background_sky_map = scaled_array.ScaledSquarePixelArray(array=6.0 * np.ones((6, 6)), pixel_scale=3.0)

    return ccd.CCDData(image=image, pixel_scale=3.0, psf=psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=exposure_time_map, background_sky_map=background_sky_map)

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True, True, True],
                              [True, True, True, True, True, True],
                              [True, True, False, False, True, True],
                              [True, True, False, False, True, True],
                              [True, True, True, True, True, True],
                              [True, True, True, True, True, True]]), pixel_scale=3.0)

@pytest.fixture(name="lens_data")
def make_lens_ccd(ccd, mask):
    return ld.LensData(ccd_data=ccd, mask=mask, optimal_sub_grid=True)


class TestLensData(object):

    def test__attributes(self, ccd, lens_data):

        assert lens_data.pixel_scale == ccd.pixel_scale
        assert lens_data.pixel_scale == 3.0

        assert (lens_data.image == ccd.image).all()
        assert (lens_data.image == np.ones((6,6))).all()

        assert (lens_data.psf == ccd.psf).all()
        assert (lens_data.psf == np.ones((3,3))).all()

        assert (lens_data.noise_map == ccd.noise_map).all()
        assert (lens_data.noise_map == 2.0*np.ones((6,6))).all()

        assert lens_data.image_psf_shape == (3,3)
        assert lens_data.inversion_psf_shape == (3,3)

    def test__masking(self, lens_data):

        assert (lens_data.image_1d == np.ones(4)).all()
        assert (lens_data.noise_map_1d == 2.0*np.ones(4)).all()
        assert (lens_data.mask_1d == np.array([False, False, False, False])).all()

    def test__grid_stack(self, lens_data):

        assert (lens_data.grid_stack.regular == np.array([[1.5, -1.5], [1.5, 1.5], [-1.5, -1.5], [-1.5, 1.5]])).all()
        assert (lens_data.grid_stack.sub == np.array([[2.25, -2.25], [2.25, -0.75], [0.75, -2.25], [0.75, -0.75],
                                                     [2.25, 0.75], [2.25, 2.25], [0.75, 0.75], [0.75, 2.25],
                                                     [-0.75, -2.25], [-0.75, -0.75], [-2.25, -2.25], [-2.25, -0.75],
                                                     [-0.75, 0.75], [-0.75, 2.25], [-2.25, 0.75], [-2.25, 2.25]])).all()
        assert (lens_data.grid_stack.blurring == np.array([[4.5, -4.5], [4.5, -1.5], [4.5, 1.5], [4.5, 4.5],
                                                          [1.5, -4.5], [1.5, 4.5], [-1.5, -4.5], [-1.5, 4.5],
                                                          [-4.5, -4.5], [-4.5, -1.5], [-4.5, 1.5], [-4.5, 4.5]])).all()

    def test__padded_grid_stack(self, lens_data):

        padded_image_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full((8, 8), False),
                                                                        pixel_scales=lens_data.image.pixel_scales)

        assert (lens_data.padded_grid_stack.regular == padded_image_util).all()
        assert lens_data.padded_grid_stack.regular.image_shape == (6, 6)
        assert lens_data.padded_grid_stack.regular.padded_shape == (8, 8)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_optimal_spacing(
            mask=np.full((8, 8), False), pixel_scales=lens_data.image.pixel_scales,
            sub_grid_size=lens_data.grid_stack.sub.sub_grid_size)

        assert lens_data.padded_grid_stack.sub == pytest.approx(padded_sub_util, 1e-4)
        assert lens_data.padded_grid_stack.sub.image_shape == (6, 6)
        assert lens_data.padded_grid_stack.sub.padded_shape == (8, 8)

        assert (lens_data.padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()

    def test__interp_pixel_scale_input__grid_stack_and_padded_grid_stack_include_interpolators(self, ccd, mask):

        lens_data = ld.LensData(ccd_data=ccd, mask=mask, interp_pixel_scale=1.0)

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
        assert (lens_data.border == np.array([0, 1, 2, 3])).all()

    def test__convolvers(self, lens_data):
        assert type(lens_data.convolver_image) == convolution.ConvolverImage
        assert type(lens_data.convolver_mapping_matrix) == inversion_convolution.ConvolverMappingMatrix

    def test__constructor_inputs(self):

        psf = ccd.PSF(np.ones((7, 7)), 1)
        image = ccd.CCDData(np.ones((51, 51)), pixel_scale=3., psf=psf, noise_map=np.ones((51, 51)))
        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(51, 51), pixel_scale=1.0, invert=True)
        mask[26, 26] = False

        lens_data = ld.LensData(image, mask, sub_grid_size=8, image_psf_shape=(5, 5),
                                inversion_psf_shape=(3, 3), positions=[np.array([[1.0, 1.0]])])

        assert lens_data.sub_grid_size == 8
        assert lens_data.convolver_image.psf_shape == (5, 5)
        assert lens_data.convolver_mapping_matrix.psf_shape == (3, 3)
        assert (lens_data.positions[0] == np.array([[1.0, 1.0]])).all()

        assert lens_data.image_psf_shape == (5,5)
        assert lens_data.inversion_psf_shape == (3, 3)

    def test__lens_data_with_modified_image(self, lens_data):

        lens_data = lens_data.new_lens_data_with_modified_image(modified_image=8.0 * np.ones((6, 6)))

        assert (lens_data.image == 8.0*np.ones((6,6))).all()
        assert (lens_data.image_1d == 8.0*np.ones(4)).all()

    def test__lens_data_with_binned_up_ccd_data(self, lens_data):

        binned_up_psf = lens_data.ccd_data.psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5)

        lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(bin_up_factor=2)

        assert (lens_data.image == np.ones((3,3))).all()
        assert (lens_data.psf == binned_up_psf).all()
        assert (lens_data.noise_map == np.ones((3,3))).all()
        assert (lens_data.ccd_data.background_noise_map == (np.sqrt(36)/4.0)*np.ones((3,3))).all()
        assert (lens_data.ccd_data.poisson_noise_map == (np.sqrt(64)/4.0)*np.ones((3,3))).all()
        assert (lens_data.ccd_data.exposure_time_map == 20.0*np.ones((3,3))).all()
        assert (lens_data.ccd_data.background_sky_map == 6.0*np.ones((3,3))).all()

        assert (lens_data.mask == np.array([[True, True, True],
                                            [True, False, True],
                                            [True, True, True]])).all()

        assert (lens_data.image_1d == np.ones((1))).all()
        assert (lens_data.noise_map_1d == np.ones((1))).all()

@pytest.fixture(name="lens_data_hyper")
def make_lens_hyper_image(ccd, mask):

    return ld.LensDataHyper(ccd_data=ccd, mask=mask, hyper_model_image=10.0 * np.ones((6, 6)),
                            hyper_galaxy_images=[11.0*np.ones((6,6)), 12.0*np.ones((6,6))],
                            hyper_minimum_values=[0.1, 0.2])


class TestLensDataHyper(object):

    def test__attributes(self, ccd, lens_data_hyper):

        assert lens_data_hyper.pixel_scale == ccd.pixel_scale

        assert (lens_data_hyper.image == ccd.image).all()
        assert (lens_data_hyper.image == np.ones((6,6))).all()

        assert (lens_data_hyper.psf == ccd.psf).all()
        assert (lens_data_hyper.psf == np.ones((3,3))).all()

        assert (lens_data_hyper.noise_map == ccd.noise_map).all()
        assert (lens_data_hyper.noise_map == 2.0*np.ones((6,6))).all()

        assert (lens_data_hyper.hyper_model_image == 10.0*np.ones((6,6))).all()
        assert (lens_data_hyper.hyper_galaxy_images[0] == 11.0*np.ones((6,6))).all()
        assert (lens_data_hyper.hyper_galaxy_images[1] == 12.0*np.ones((6,6))).all()

        assert lens_data_hyper.hyper_minimum_values == [0.1, 0.2]

    def test__masking(self, lens_data_hyper):

        assert (lens_data_hyper.image_1d == np.ones(4)).all()
        assert (lens_data_hyper.noise_map_1d == 2.0*np.ones(4)).all()

        assert (lens_data_hyper.hyper_model_image_1d == 10.0*np.ones(4)).all()
        assert (lens_data_hyper.hyper_galaxy_images_1d[0] == 11.0*np.ones(4)).all()
        assert (lens_data_hyper.hyper_galaxy_images_1d[1] == 12.0*np.ones(4)).all()
