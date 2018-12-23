import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens.data.array import mask as msk
from autolens.data.array import grids
from autolens.data.imaging import image as im
from autolens.data.imaging import convolution
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.util import lens_fit_stack_util as stack_util
from autolens.lens.stack import ray_tracing_stack
from autolens.model.galaxy import galaxy as g

@pytest.fixture(name='mask_0')
def make_mask_0():
    return msk.Mask(array=np.array([[True, True, True, True],
                                     [True, False, False, True],
                                     [True, False, False, True],
                                     [True, True, True, True]]), pixel_scale=1.0)

@pytest.fixture(name='mask_1')
def make_mask_1():
    return msk.Mask(array=np.array([[True, True, True, True],
                                     [True, False, False, True],
                                     [True, True, False, True],
                                     [True, True, True, True]]), pixel_scale=1.0)

@pytest.fixture(name='blurring_mask_0')
def make_blurring_mask_0():
    return msk.Mask(array=np.array([[False, False, False, False],
                                     [False, True, True, False],
                                     [False, True, True, False],
                                     [False, False, False, False]]), pixel_scale=1.0)

@pytest.fixture(name='blurring_mask_1')
def make_blurring_mask_1():
    return msk.Mask(array=np.array([[False, False, False, False],
                                     [False, True, True, False],
                                     [False, True, False, False],
                                     [False, False, False, True]]), pixel_scale=1.0)

@pytest.fixture(name='convolver_no_blur_0')
def make_convolver_no_blur_0(mask_0, blurring_mask_0):

    psf = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])

    return convolution.ConvolverImage(mask=mask_0, blurring_mask=blurring_mask_0, psf=psf)

@pytest.fixture(name='convolver_no_blur_1')
def make_convolver_no_blur_1(mask_1, blurring_mask_1):

    psf = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])

    return convolution.ConvolverImage(mask=mask_1, blurring_mask=blurring_mask_1, psf=psf)

@pytest.fixture(name='convolver_blur_0')
def make_convolver_blur_0(mask_0, blurring_mask_0):

    psf = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

    return convolution.ConvolverImage(mask=mask_0, blurring_mask=blurring_mask_0, psf=psf)

@pytest.fixture(name='convolver_blur_1')
def make_convolver_blur_1(mask_1, blurring_mask_1):

    psf = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

    return convolution.ConvolverImage(mask=mask_1, blurring_mask=blurring_mask_1, psf=psf)

@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))


class MapArrays1D:

    def test__two_1d_arrays_in__mapped_2d_arrays_out(self, mask_0, mask_1):

        array_1d_0 = np.array([1.0, 2.0, 3.0, 4.0])
        regular_grid_0 = grids.RegularGrid.from_mask(mask=mask_0)

        array_1d_1 = np.array([5.0, 6.0, 7.0])
        regular_grid_1 = grids.RegularGrid.from_mask(mask=mask_1)

        arrays_2d = stack_util.map_arrays_1d_to_scaled_arrays(arrays_1d=[array_1d_0, array_1d_1],
                                  map_to_scaled_arrays=[regular_grid_0.scaled_array_from_array_1d,
                                                        regular_grid_1.scaled_array_from_array_1d])

        assert (arrays_2d[0] == np.array([[0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 2.0, 0.0],
                                          [0.0, 3.0, 4.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0]])).all()

        assert (arrays_2d[1] == np.array([[0.0, 0.0, 0.0, 0.0],
                                          [0.0, 5.0, 6.0, 0.0],
                                          [0.0, 7.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0]])).all()


class TestBlurredImages:

    def test__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, convolver_no_blur_0, convolver_blur_0):

        image_1d_0 = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        image_1d_1 = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_images_1d = stack_util.blurred_images_1d_of_images_from_1d_unblurred_and_bluring_images(
            unblurred_images_1d=[image_1d_0, image_1d_1],
            blurring_images_1d=[blurring_image_1d_0, blurring_image_1d_1],
            convolvers=[convolver_no_blur_0, convolver_blur_0])

        assert (blurred_images_1d[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (blurred_images_1d[1] == np.array([4.0, 4.0, 4.0, 4.0])).all()


class TestBlurredImageOfPlanes:

    def test__blurred_image_of_planes__x2_images(self, mask_0, mask_1, convolver_blur_0, convolver_blur_1):

        data_grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask_0,
                                                sub_grid_size=1, psf_shape=(3, 3))

        data_grid_stack_1 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask_1,
                                                sub_grid_size=1, psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stacks=[data_grid_stack_0, data_grid_stack_1])

        blurred_lens_image_0 = convolver_blur_0.convolve_image(image_array=tracer.image_plane.image_plane_images_1d[0],
                                                           blurring_array=tracer.image_plane.image_plane_blurring_images_1d[0])
        blurred_lens_image_0 = data_grid_stack_0.regular.scaled_array_from_array_1d(array_1d=blurred_lens_image_0)

        blurred_source_image_0 = convolver_blur_0.convolve_image(image_array=tracer.source_plane.image_plane_images_1d[0],
                                                             blurring_array=tracer.source_plane.image_plane_blurring_images_1d[0])
        blurred_source_image_0 = data_grid_stack_0.regular.scaled_array_from_array_1d(array_1d=blurred_source_image_0)

        blurred_lens_image_1 = convolver_blur_1.convolve_image(image_array=tracer.image_plane.image_plane_images_1d[1],
                                                           blurring_array=tracer.image_plane.image_plane_blurring_images_1d[1])
        blurred_lens_image_1 = data_grid_stack_1.regular.scaled_array_from_array_1d(array_1d=blurred_lens_image_1)

        blurred_source_image_1 = convolver_blur_1.convolve_image(image_array=tracer.source_plane.image_plane_images_1d[1],
                                                             blurring_array=tracer.source_plane.image_plane_blurring_images_1d[1])
        blurred_source_image_1 = data_grid_stack_1.regular.scaled_array_from_array_1d(array_1d=blurred_source_image_1)

        blurred_images_of_planes = stack_util.blurred_images_of_images_and_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_images_1d_of_planes=tracer.image_plane_images_1d_of_planes,
            image_plane_blurring_images_1d_of_planes=tracer.image_plane_blurring_images_1d_of_planes,
            convolvers=[convolver_blur_0, convolver_blur_1],
            map_to_scaled_arrays=[data_grid_stack_0.regular.array_2d_from_array_1d,
                                  data_grid_stack_1.regular.array_2d_from_array_1d])

        assert (blurred_images_of_planes[0][0] == blurred_lens_image_0).all()
        assert (blurred_images_of_planes[0][1] == blurred_source_image_0).all()
        assert (blurred_images_of_planes[1][0] == blurred_lens_image_1).all()
        assert (blurred_images_of_planes[1][1] == blurred_source_image_1).all()


class TestUnmaskedModelImages:

    def test___3x3_padded_image__one_psf_blurs__other_asymmetric(self):

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        psf_0 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        psf_1 = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                        [0.0, 1.0, 2.0],
                                        [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        unmasked_image_1d_1 = np.zeros(25)
        unmasked_image_1d_1[12] = 1.0

        unmasked_blurred_images = stack_util.unmasked_blurred_images_from_padded_grid_stacks_psfs_and_unmasked_images(
            padded_grid_stacks=[padded_grid_stack, padded_grid_stack], psfs=[psf_0, psf_1],
            unmasked_images_1d=[np.ones(25), unmasked_image_1d_1])

        assert (unmasked_blurred_images[0] == np.ones((3, 3))).all()
        assert (unmasked_blurred_images[1] == np.array([[0.0, 3.0, 0.0],
                                                        [0.0, 1.0, 2.0],
                                                        [0.0, 0.0, 0.0]])).all()


class TestUnmaskedModelImageOfGalaxies:

    def test___x2_galaxies__3x3_padded_image__x2_images__one_blurs__one_asymetric_psf_blurring(self):

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        psf_0 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        psf_1 = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                        [0.0, 1.0, 2.0],
                                        [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))

        tracer = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0, g1],
                                                         image_plane_grid_stacks=[padded_grid_stack, padded_grid_stack])

        manual_blurred_image_00 = tracer.image_plane.image_plane_images_1d_of_galaxies[0][0]
        manual_blurred_image_00 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_00)
        manual_blurred_image_00 = psf_0.convolve(array=manual_blurred_image_00)

        manual_blurred_image_01 = tracer.image_plane.image_plane_images_1d_of_galaxies[0][1]
        manual_blurred_image_01 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_01)
        manual_blurred_image_01 = psf_0.convolve(array=manual_blurred_image_01)

        manual_blurred_image_10 = tracer.image_plane.image_plane_images_1d_of_galaxies[1][0]
        manual_blurred_image_10 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_10)
        manual_blurred_image_10 = psf_1.convolve(array=manual_blurred_image_10)

        manual_blurred_image_11 = tracer.image_plane.image_plane_images_1d_of_galaxies[1][1]
        manual_blurred_image_11 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_11)
        manual_blurred_image_11 = psf_1.convolve(array=manual_blurred_image_11)

        unmasked_blurred_image_of_images_planes_and_galaxies = \
            stack_util.unmasked_blurred_images_of_images_planes_and_galaxies_from_padded_grid_stacks_and_psf(
                planes=tracer.planes, padded_grid_stacks=[padded_grid_stack, padded_grid_stack], psfs=[psf_0, psf_1])

        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][0][0] == manual_blurred_image_00[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][0][1] == manual_blurred_image_01[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][0][0] == manual_blurred_image_10[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][0][1] == manual_blurred_image_11[1:4, 1:4]).all()
        
    def test___same_as_above__image_and_source_plane(self):

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        psf_0 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        psf_1 = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                        [0.0, 1.0, 2.0],
                                        [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

        tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0, g1, g2], source_galaxies=[g0, g1],
                                              image_plane_grid_stacks=[padded_grid_stack, padded_grid_stack])

        manual_blurred_image_i00 = tracer.image_plane.image_plane_images_1d_of_galaxies[0][0]
        manual_blurred_image_i00 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_i00)
        manual_blurred_image_i00 = psf_0.convolve(array=manual_blurred_image_i00)

        manual_blurred_image_i01 = tracer.image_plane.image_plane_images_1d_of_galaxies[0][1]
        manual_blurred_image_i01 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_i01)
        manual_blurred_image_i01 = psf_0.convolve(array=manual_blurred_image_i01)

        manual_blurred_image_i10 = tracer.image_plane.image_plane_images_1d_of_galaxies[1][0]
        manual_blurred_image_i10 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_i10)
        manual_blurred_image_i10 = psf_1.convolve(array=manual_blurred_image_i10)

        manual_blurred_image_i11 = tracer.image_plane.image_plane_images_1d_of_galaxies[1][1]
        manual_blurred_image_i11 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_i11)
        manual_blurred_image_i11 = psf_1.convolve(array=manual_blurred_image_i11)
        

        manual_blurred_image_s00 = tracer.source_plane.image_plane_images_1d_of_galaxies[0][0]
        manual_blurred_image_s00 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_s00)
        manual_blurred_image_s00 = psf_0.convolve(array=manual_blurred_image_s00)

        manual_blurred_image_s01 = tracer.source_plane.image_plane_images_1d_of_galaxies[0][1]
        manual_blurred_image_s01 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_s01)
        manual_blurred_image_s01 = psf_0.convolve(array=manual_blurred_image_s01)

        manual_blurred_image_s10 = tracer.source_plane.image_plane_images_1d_of_galaxies[1][0]
        manual_blurred_image_s10 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_s10)
        manual_blurred_image_s10 = psf_1.convolve(array=manual_blurred_image_s10)

        manual_blurred_image_s11 = tracer.source_plane.image_plane_images_1d_of_galaxies[1][1]
        manual_blurred_image_s11 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_s11)
        manual_blurred_image_s11 = psf_1.convolve(array=manual_blurred_image_s11)

        unmasked_blurred_image_of_images_planes_and_galaxies = \
            stack_util.unmasked_blurred_images_of_images_planes_and_galaxies_from_padded_grid_stacks_and_psf(
                planes=tracer.planes, padded_grid_stacks=[padded_grid_stack, padded_grid_stack], psfs=[psf_0, psf_1])

        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][0][0] == manual_blurred_image_i00[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][0][1] == manual_blurred_image_i01[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][0][0] == manual_blurred_image_i10[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][0][1] == manual_blurred_image_i11[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][1][0] == manual_blurred_image_s00[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[0][1][1] == manual_blurred_image_s01[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][1][0] == manual_blurred_image_s10[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_images_planes_and_galaxies[1][1][1] == manual_blurred_image_s11[1:4, 1:4]).all()