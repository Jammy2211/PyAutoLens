import numpy as np
import pytest

from autolens.imaging import image
from autolens.imaging import mask as mask
from autolens.inversion import inversions
from autolens.inversion import pixelizations
from autolens.inversion import regularization
from autolens.lensing import lensing_fitting
from autolens.galaxy import galaxy as g
from autolens.fitting import fitting
from autolens.lensing import lensing_image
from autolens.lensing import plane as pl
from autolens.lensing import ray_tracing
from autolens.profiles import light_profiles as lp
from test.mock.mock_profiles import MockLightProfile
from test.mock.mock_galaxy import MockHyperGalaxy
from test.mock.mock_lensing import MockTracer


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [g.Galaxy()]


@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return g.Galaxy(light_profile=sersic)


@pytest.fixture(name='li_no_blur')
def make_li_no_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)


@pytest.fixture(name='li_blur')
def make_li_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)


@pytest.fixture(name='li_no_blur_1x1')
def make_li_no_blur_1x1():
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)

    im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

    ma = mask.Mask(array=np.array([[True, True, True],
                                   [True, False, True],
                                   [True, True, True]]), pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=1)


@pytest.fixture(name='li_manual')
def make_li_manual():
    im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 5.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                     [2.0, 5.0, 1.0],
                                     [3.0, 4.0, 0.0]])), pixel_scale=1.0)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
    ma = mask.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=1)


@pytest.fixture(name='hyper')
def make_hyper():
    class Hyper(object):

        def __init__(self):
            pass

    hyper = Hyper()

    hyper.hyper_model_image = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])
    hyper.hyper_galaxy_images = [np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0]),
                                 np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])]
    hyper.hyper_minimum_values = [0.2, 0.8]
    return hyper

@pytest.fixture(name='li_hyper_no_blur')
def make_li_hyper_no_blur(hyper):
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingHyperImage(im, ma, hyper_model_image=hyper.hyper_model_image,
                                           hyper_galaxy_images=hyper.hyper_galaxy_images,
                                           hyper_minimum_values=hyper.hyper_minimum_values, sub_grid_size=1)


@pytest.fixture(name='li_hyper_no_blur_1x1')
def make_li_hyper_no_blur_1x1(hyper):
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)

    im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

    ma = mask.Mask(array=np.array([[True, True, True],
                                   [True, False, True],
                                   [True, True, True]]), pixel_scale=1.0)

    return lensing_image.LensingHyperImage(im, ma, hyper_model_image=hyper.hyper_model_image,
                                           hyper_galaxy_images=hyper.hyper_galaxy_images,
                                           hyper_minimum_values=hyper.hyper_minimum_values, sub_grid_size=1)

@pytest.fixture(name='li_hyper_manual')
def make_li_hyper_manual(hyper):
    im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 5.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                     [2.0, 5.0, 1.0],
                                     [3.0, 4.0, 0.0]])), pixel_scale=1.0)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
    ma = mask.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    return lensing_image.LensingHyperImage(im, ma, hyper_model_image=hyper.hyper_model_image,
                                           hyper_galaxy_images=hyper.hyper_galaxy_images,
                                           hyper_minimum_values=hyper.hyper_minimum_values, sub_grid_size=1)


class TestUnmaskedModelImages:

    def test___of_galaxies__x1_galaxy__3x3_padded_image__no_psf_blurring(self, galaxy_light):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.padded_grids)

        manual_model_image_0 = tracer.image_plane.grids.image.map_to_2d_keep_padded(tracer._image_plane_image)
        manual_model_image_0 = psf.convolve(manual_model_image_0)

        padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li,
                                                                                                              tracer)

        assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()

    def test___of_galaxies__x1_galaxy__3x3_padded_image__asymetric_psf_blurring(self, galaxy_light):
        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.padded_grids)

        manual_model_image_0 = tracer.image_plane.grids.image.map_to_2d_keep_padded(tracer._image_plane_image)
        manual_model_image_0 = psf.convolve(manual_model_image_0)

        padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li, tracer)

        assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()

    def test___of_galaxies__x2_galaxies__3x3_padded_image__asymetric_psf_blurring(self):
        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grids=li.padded_grids)
        padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li, tracer)

        manual_model_image_0 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.image_plane._image_plane_image_of_galaxies[0])
        manual_model_image_0 = psf.convolve(manual_model_image_0)

        manual_model_image_1 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.image_plane._image_plane_image_of_galaxies[1])
        manual_model_image_1 = psf.convolve(manual_model_image_1)

        assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
        assert (manual_model_image_1[1:4, 1:4] == padded_model_images[0][1]).all()

    def test___same_as_above_but_image_and_souce_plane(self):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grids=li.padded_grids)

        manual_model_image_0 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.image_plane._image_plane_image_of_galaxies[0])
        manual_model_image_0 = psf.convolve(manual_model_image_0)

        manual_model_image_1 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.image_plane._image_plane_image_of_galaxies[1])
        manual_model_image_1 = psf.convolve(manual_model_image_1)


        manual_model_image_2 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.source_plane._image_plane_image_of_galaxies[0])
        manual_model_image_2 = psf.convolve(manual_model_image_2)

        manual_model_image_3 = tracer.image_plane.grids.image.map_to_2d_keep_padded(
            tracer.source_plane._image_plane_image_of_galaxies[1])
        manual_model_image_3 = psf.convolve(manual_model_image_3)

        padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li, tracer)

        assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
        assert (manual_model_image_1[1:4, 1:4] == padded_model_images[0][1]).all()
        assert (manual_model_image_2[1:4, 1:4] == padded_model_images[1][0]).all()
        assert (manual_model_image_3[1:4, 1:4] == padded_model_images[1][1]).all()

    def test__properties_of_fit(self, galaxy_light):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        padded_tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.padded_grids)

        manual_model_image_0 = \
            padded_tracer.image_plane.grids.image.map_to_2d_keep_padded(padded_tracer._image_plane_image)
        manual_model_image_0 = psf.convolve(manual_model_image_0)

        padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li,
                                                                                            padded_tracer)

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.grids)

        fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer, padded_tracer=padded_tracer)

        assert (manual_model_image_0[1:4, 1:4] == fit.unmasked_model_profile_image).all()
        assert (padded_model_images[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0]).all()

    def test__padded_tracer_is_none__returns_none(self, galaxy_light):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)
        li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.grids)

        fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer, padded_tracer=None)

        assert fit.unmasked_model_profile_image == None
        assert fit.unmasked_model_profile_images_of_galaxies == None


class TestLensingProfileFit:

    class TestModelImages:

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__blurring_region__image_blurs_to_9s(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                has_light_profile=True, has_hyper_galaxy=False, has_pixelization=False)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li_blur, tracer=tracer)

            assert (fit.model_image == np.array([[0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 9.0, 9.0, 0.0],
                                                 [0.0, 9.0, 9.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0]])).all()

        def test__real_tracer__2x2_image__no_psf_blurring(self, li_no_blur, galaxy_light):

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li_no_blur.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li_no_blur, tracer=tracer)

            tracer_image = tracer._image_plane_image
            assert (tracer_image == fit._model_image).all()

            tracer_image_2d = li_no_blur.grids.image.scaled_array_from_array_1d(tracer_image)
            assert (tracer_image_2d == fit.model_image).all()

        def test__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self, galaxy_light):
            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

            central_values = tracer._image_plane_image
            blurring_values = tracer._image_plane_blurring_image

            tracer_blurred_image_manual_0 = 2.0 * central_values[0] + 3.0 * central_values[2] + blurring_values[4]
            tracer_blurred_image_manual_1 = 2.0 * central_values[1] + 3.0 * central_values[3] + central_values[0]
            tracer_blurred_image_manual_2 = 2.0 * central_values[2] + 3.0 * blurring_values[9] + blurring_values[6]
            tracer_blurred_image_manual_3 = 2.0 * central_values[3] + 3.0 * blurring_values[10] + central_values[2]

            assert tracer_blurred_image_manual_0 == pytest.approx(fit._model_image[0], 1e-6)
            assert tracer_blurred_image_manual_1 == pytest.approx(fit._model_image[1], 1e-6)
            assert tracer_blurred_image_manual_2 == pytest.approx(fit._model_image[2], 1e-6)
            assert tracer_blurred_image_manual_3 == pytest.approx(fit._model_image[3], 1e-6)

            assert tracer_blurred_image_manual_0 == pytest.approx(fit.model_image[1, 1], 1e-6)
            assert tracer_blurred_image_manual_1 == pytest.approx(fit.model_image[1, 2], 1e-6)
            assert tracer_blurred_image_manual_2 == pytest.approx(fit.model_image[2, 1], 1e-6)
            assert tracer_blurred_image_manual_3 == pytest.approx(fit.model_image[2, 2], 1e-6)

        def test__model_images_of_planes__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            model_lens_image = li.convolver_image.convolve_image(tracer.image_plane._image_plane_image,
                                                                 tracer.image_plane._image_plane_blurring_image)

            model_source_image = li.convolver_image.convolve_image(tracer.source_plane._image_plane_image,
                                                                   tracer.source_plane._image_plane_blurring_image)

            model_lens_image = li.grids.image.scaled_array_from_array_1d(model_lens_image)
            model_source_image = li.grids.image.scaled_array_from_array_1d(model_source_image)

            assert (fit.model_images_of_planes[0] == model_lens_image).all()
            assert (fit.model_images_of_planes[1] == model_source_image).all()

        def test__same_as_above_but_multi_tracer(self):
            
            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), redshift=0.1)
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0), redshift=0.2)
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0), redshift=0.3)

            from astropy import cosmology as cosmo

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=li.grids,
                                             cosmology=cosmo.Planck15)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            model_plane_image_0 = li.convolver_image.convolve_image(tracer.planes[0]._image_plane_image,
                                                                 tracer.planes[0]._image_plane_blurring_image)

            model_plane_image_1 = li.convolver_image.convolve_image(tracer.planes[1]._image_plane_image,
                                                                 tracer.planes[1]._image_plane_blurring_image)
            
            model_plane_image_2 = li.convolver_image.convolve_image(tracer.planes[2]._image_plane_image,
                                                                 tracer.planes[2]._image_plane_blurring_image)

            model_plane_image_0 = li.grids.image.scaled_array_from_array_1d(model_plane_image_0)
            model_plane_image_1 = li.grids.image.scaled_array_from_array_1d(model_plane_image_1)
            model_plane_image_2 = li.grids.image.scaled_array_from_array_1d(model_plane_image_2)

            assert (fit.model_images_of_planes[0] == model_plane_image_0).all()
            assert (fit.model_images_of_planes[1] == model_plane_image_1).all()
            assert (fit.model_images_of_planes[2] == model_plane_image_2).all()

        def test__model_images_of_planes__if_galaxy_has_no_light_profile__replace_with_none(self):
            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))
            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g0_image = pl.intensities_from_grid(grid=li.grids.sub, galaxies=[g0])
            g0_blurring_image = pl.intensities_from_grid(grid=li.grids.blurring, galaxies=[g0])
            g0_model_image = li.grids.image.scaled_array_from_array_1d(li.convolver_image.convolve_image(g0_image, g0_blurring_image))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert (fit.model_images_of_planes[0] == g0_model_image).all()
            assert (fit.model_images_of_planes[1] == g0_model_image).all()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy()],
                                                         image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert (fit.model_images_of_planes[0] == g0_model_image).all()
            assert fit.model_images_of_planes[1] == None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g0],
                                                         image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert fit.model_images_of_planes[0] == None
            assert (fit.model_images_of_planes[1] == g0_model_image).all()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g.Galaxy()],
                                                         image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert fit.model_images_of_planes[0] == None
            assert fit.model_images_of_planes[1] == None

    class TestLikelihood:

        def test__1x1_image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_term(self):
            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

            ma = mask.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x1_image__tracing_fits_data_with_chi_sq_5(self):
            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            im = image.Image(5.0 * np.ones((3, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 4)))
            im[1,2]  = 4.0

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert fit.chi_squared_term == 25.0
            assert fit.reduced_chi_squared == 25.0 / 2.0
            assert fit.likelihood == -0.5 * (25.0 + 2.0*np.log(2 * np.pi * 1.0))

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grids=li_manual.grids)

            padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                                  image_plane_grids=li_manual.padded_grids)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_manual, tracer=tracer,
                                                                padded_tracer=padded_tracer)

            image_im = tracer._image_plane_image
            blurring_im = tracer._image_plane_blurring_image
            model_image = li_manual.convolver_image.convolve_image(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(li_manual, model_image)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_manual.noise_map)

            assert li_manual.grids.image.scaled_array_from_array_1d(li_manual.noise_map) == pytest.approx(fit.noise_map, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(model_image) == pytest.approx(fit.model_image, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(residuals) == pytest.approx(fit.residuals, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(chi_squareds) == pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(li_manual.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                            tracer=tracer)
            assert fast_likelihood == pytest.approx(fit.likelihood)

            padded_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=li_manual,
                                 _unmasked_image=padded_tracer._image_plane_image)
            padded_model_image_of_galaxies = \
                lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li_manual, padded_tracer)

            assert (padded_model_image == fit.unmasked_model_profile_image).all()
            assert (padded_model_image_of_galaxies[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0]).all()
            assert (padded_model_image_of_galaxies[1][0] == fit.unmasked_model_profile_images_of_galaxies[1][0]).all()


class TestHyperLensingProfileFit:

    class TestScaledLikelihood:

        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_is_0(self, li_hyper_no_blur_1x1):
            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            g1 = g.Galaxy(light_profile=MockLightProfile(value=0.0))

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1],
                                                  image_plane_grids=li_hyper_no_blur_1x1.grids)

            li_hyper_no_blur_1x1.hyper_model_image = np.array([1.0])
            li_hyper_no_blur_1x1.hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]

            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                          noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                          noise_power=1.0)

            fit = lensing_fitting.HyperLensingProfileFit(lensing_hyper_image=li_hyper_no_blur_1x1, tracer=tracer)

            chi_squared_term = 0.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)

            assert fit.scaled_likelihood == -0.5 * (chi_squared_term + scaled_noise_term)

        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_nonzero(self, no_galaxies,
                                                                                        li_hyper_no_blur_1x1):
            li_hyper_no_blur_1x1[0] = 2.0

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            g1 = g.Galaxy(light_profile=MockLightProfile(value=0.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=no_galaxies,
                                                         image_plane_grids=li_hyper_no_blur_1x1.grids)

            li_hyper_no_blur_1x1.hyper_model_image = np.array([1.0])
            li_hyper_no_blur_1x1.hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]

            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                          noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                          noise_power=1.0)

            fit = lensing_fitting.HyperLensingProfileFit(lensing_hyper_image=li_hyper_no_blur_1x1, tracer=tracer)

            scaled_chi_squared_term = (1.0 / (4.0)) ** 2.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)

            assert fit.scaled_likelihood == -0.5 * (scaled_chi_squared_term + scaled_noise_term)

    class TestAbstractLogic:

        def test__logic_in_abstract_fit(self, li_hyper_no_blur, galaxy_light, hyper):
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=li_hyper_no_blur.grids)

            fit = lensing_fitting.HyperLensingProfileFit(lensing_hyper_image=li_hyper_no_blur, tracer=tracer)

            assert fit.total_inversions == 0

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grids=li_hyper_manual.grids)

            padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                                image_plane_grids=li_hyper_manual.padded_grids)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_hyper_manual, tracer=tracer,
                                                                padded_tracer=padded_tracer)

            image_im = tracer._image_plane_image
            blurring_im = tracer._image_plane_blurring_image
            model_image = li_hyper_manual.convolver_image.convolve_image(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(li_hyper_manual, model_image)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_hyper_manual.noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(model_image) == \
                   pytest.approx(fit.model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(li_hyper_manual.noise_map) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(li_hyper_manual.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            contributions = fitting.contributions_from_hyper_images_and_galaxies(li_hyper_manual.hyper_model_image,
                            li_hyper_manual.hyper_galaxy_images,[hyper_galaxy, hyper_galaxy],
                            li_hyper_manual.hyper_minimum_values)

            scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                                                                                          [hyper_galaxy,
                                                                                           hyper_galaxy],
                                                                                          li_hyper_manual.noise_map)
            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, scaled_noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[0]) == \
                   pytest.approx(fit.contributions[0], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[1]) == \
                   pytest.approx(fit.contributions[1], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_noise_map) == \
                   pytest.approx(fit.scaled_noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_chi_squareds) == \
                   pytest.approx(fit.scaled_chi_squareds, 1e-4)

            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_noise_map(scaled_noise_map)
            scaled_likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(scaled_chi_squared_term,
                                                                                            scaled_noise_term)

            assert scaled_likelihood == pytest.approx(fit.scaled_likelihood, 1e-4)

            fast_scaled_likelihood = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(li_hyper_manual,
                                                                                                   tracer)

            assert fast_scaled_likelihood == fit.scaled_likelihood

            padded_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=li_hyper_manual,
                                                                                 _unmasked_image=padded_tracer._image_plane_image)
            padded_model_image_of_galaxies = \
                lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(li_hyper_manual, padded_tracer)

            assert (padded_model_image == fit.unmasked_model_profile_image).all()
            assert (padded_model_image_of_galaxies[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0]).all()
            assert (padded_model_image_of_galaxies[1][0] == fit.unmasked_model_profile_images_of_galaxies[1][0]).all()


class TestInversionLensingFit:

    class TestAbstractLogic:

        def test__logic_in_abstract_fit(self, li_no_blur):

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li_no_blur.grids, borders=li_no_blur.borders)

            fit = lensing_fitting.LensingInversionFit(lensing_image=li_no_blur, tracer=tracer)

            assert fit.total_inversions == 1

    class TestModelImageOfPlanes:

        def test__model_images_are_none_and_an_image(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)

            ma = np.array([[True, True, True, True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, True, True, True, True]])

            ma = mask.Mask(ma, pixel_scale=1.0)

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            li = lensing_image.LensingImage(im, ma, sub_grid_size=2)

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li.grids, borders=li.borders)
            fit = lensing_fitting.LensingInversionFit(lensing_image=li, tracer=tracer)

            assert fit.model_images_of_planes[0] == None
            assert fit.model_images_of_planes[1] == pytest.approx(np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-2)

    class TestRectangularInversion:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):
            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)

            ma = np.array([[True, True, True, True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, True, True, True, True]])
            ma = mask.Mask(ma, pixel_scale=1.0)

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            li = lensing_image.LensingImage(im, ma, sub_grid_size=2)

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li.grids, borders=li.borders)
            fit = lensing_fitting.LensingInversionFit(lensing_image=li, tracer=tracer)

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            reg_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])
            reg_matrix = reg_matrix + 1e-8 * np.identity(9)
            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 1e-8
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = np.log(np.linalg.det(reg_matrix))
            noise_term = 9.0 * np.log(2 * np.pi * 1.0 ** 2.0)
            evidence = -0.5 * (chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fit.evidence == pytest.approx(evidence, 1e-4)

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))

            g0 = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g0],
                                                         image_plane_grids=li_manual.grids, borders=li_manual.borders)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_manual, tracer=tracer)

            mapper = pix.mapper_from_grids_and_borders(li_manual.grids, li_manual.borders)
            inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(mapper=mapper, regularization=reg,
                                                                                          image=li_manual,
                                                                                          noise_map=li_manual.noise_map,
                                                                                          convolver=li_manual.convolver_mapping_matrix)

            residuals = fitting.residuals_from_image_and_model(li_manual, inversion.reconstructed_data_vector)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_manual.noise_map)

            assert li_manual.grids.image.scaled_array_from_array_1d(li_manual.noise_map) == pytest.approx(fit.noise_map, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(inversion.reconstructed_data_vector) == pytest.approx(fit.model_image,
                                                                                                                          1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(residuals) == pytest.approx(fit.residuals, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(chi_squareds) == pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)

            noise_term = fitting.noise_term_from_noise_map(li_manual.noise_map)

            likelihood_with_regularization = \
                fitting.likelihood_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_term,
                                                                                                               inversion.regularization_term, noise_term)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-2)

            evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                                          inversion.log_det_curvature_reg_matrix_term,
                                                                          inversion.log_det_regularization_matrix_term,
                                                                          noise_term)

            assert evidence == fit.evidence

            fast_evidence = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                          tracer=tracer)
            assert fast_evidence == evidence


class TestHyperLensingInversionFit:

    class TestRectangularInversion:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)
            ma = np.array([[True, True, True, True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, True, True, True, True]])
            ma = mask.Mask(ma, pixel_scale=1.0)
            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            hyper_model_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            hyper_galaxy_images = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
            li = lensing_image.LensingHyperImage(im, ma, hyper_model_image=hyper_model_image,
                                                 hyper_galaxy_images=hyper_galaxy_images,
                                                 hyper_minimum_values=[0.0, 0.0],
                                                 sub_grid_size=2)

            hyper_galaxy = g.HyperGalaxy(contribution_factor=0.0, noise_factor=1.0, noise_power=1.0)

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)),
                                  hyper_galaxy=hyper_galaxy)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li.grids, borders=li.borders)

            fit = lensing_fitting.HyperLensingInversionFit(lensing_hyper_image=li, tracer=tracer)

            curvature_matrix = np.array([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])
            regularization_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                              [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                              [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                              [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                              [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                              [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                              [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                              [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])
            regularization_matrix = regularization_matrix + 1e-8 * np.identity(9)
            curvature_reg_matrix = curvature_matrix + regularization_matrix

            scaled_chi_sq_term = 0.0
            gl_term = 1e-8
            det_curvature_reg_term = np.log(np.linalg.det(curvature_reg_matrix))
            det_regularization_term = np.log(np.linalg.det(regularization_matrix))
            scaled_noise_term = 9.0 * np.log(2 * np.pi * 2.0 ** 2.0)

            scaled_evidence = -0.5 * (scaled_chi_sq_term + gl_term + det_curvature_reg_term - det_regularization_term +
                                      scaled_noise_term)

            assert fit.scaled_evidence == pytest.approx(scaled_evidence, 1e-4)

    class TestAbstractLogic:

        def test__logic_in_abstract_fit(self, li_hyper_no_blur):
            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li_hyper_no_blur.grids,
                                                         borders=li_hyper_no_blur.borders)

            fit = lensing_fitting.HyperLensingInversionFit(lensing_hyper_image=li_hyper_no_blur, tracer=tracer)

            assert fit.total_inversions == 1

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            mapper = pix.mapper_from_grids_and_borders(li_hyper_manual.grids, li_hyper_manual.borders)
            reg = regularization.Constant(coefficients=(1.0,))

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            hyp_galaxy = g.Galaxy(hyper_galaxy=hyper_galaxy)
            inv_galaxy = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[hyp_galaxy, hyp_galaxy],
                                                         source_galaxies=[inv_galaxy],
                                                         image_plane_grids=li_hyper_manual.grids, 
                                                         borders=li_hyper_manual.borders)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_hyper_manual, tracer=tracer)

            inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(mapper=mapper, regularization=reg,
                        image=li_hyper_manual, noise_map=li_hyper_manual.noise_map, 
                        convolver=li_hyper_manual.convolver_mapping_matrix)

            residuals = fitting.residuals_from_image_and_model(li_hyper_manual, inversion.reconstructed_data_vector)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_hyper_manual.noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(li_hyper_manual.noise_map) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(inversion.reconstructed_data_vector) == \
                   pytest.approx(fit.model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(li_hyper_manual.noise_map)

            likelihood_with_regularization = \
                fitting.likelihood_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_term,
                                                                              inversion.regularization_term, noise_term)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-2)

            evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                                  inversion.log_det_curvature_reg_matrix_term,
                                                                  inversion.log_det_regularization_matrix_term,
                                                                  noise_term)

            assert evidence == fit.evidence

            contributions = fitting.contributions_from_hyper_images_and_galaxies(li_hyper_manual.hyper_model_image,
                            li_hyper_manual.hyper_galaxy_images, [hyper_galaxy, hyper_galaxy],
                            li_hyper_manual.hyper_minimum_values)
            scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                                                                                          [hyper_galaxy,
                                                                                           hyper_galaxy],
                                                                                          li_hyper_manual.noise_map)

            scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(mapper=mapper,
                               regularization=reg, image=li_hyper_manual, noise_map=scaled_noise_map,
                               convolver=li_hyper_manual.convolver_mapping_matrix)

            scaled_model_image = scaled_inversion.reconstructed_data_vector
            scaled_residuals = fitting.residuals_from_image_and_model(li_hyper_manual, scaled_inversion.reconstructed_data_vector)
            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(scaled_residuals, scaled_noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[0]) == \
                   pytest.approx(fit.contributions[0], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[1]) == \
                   pytest.approx(fit.contributions[1], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_noise_map) == \
                   pytest.approx(fit.scaled_noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_model_image) == \
                   pytest.approx(fit.scaled_model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_residuals) == \
                   pytest.approx(fit.scaled_residuals, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_chi_squareds) == \
                   pytest.approx(fit.scaled_chi_squareds, 1e-4)

            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_noise_map(scaled_noise_map)
            scaled_evidence = fitting.evidence_from_reconstruction_terms(scaled_chi_squared_term,
                              scaled_inversion.regularization_term, scaled_inversion.log_det_curvature_reg_matrix_term,
                              scaled_inversion.log_det_regularization_matrix_term, scaled_noise_term)
            assert scaled_evidence == fit.scaled_evidence

            fast_scaled_evidence = \
                lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=li_hyper_manual,
                                                                              tracer=tracer)
            assert fast_scaled_evidence == scaled_evidence


class TestLensingProfileInversionFit:

    class TestModelImagesOfPLanes:

        def test___model_images_of_planes_are_profile_and_inversion_images(self, li_manual):

            galaxy_light = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))
            galaxy_pix = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li_manual.grids, borders=li_manual.borders)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_manual, tracer=tracer)

            assert (fit.model_images_of_planes[0] == fit.profile_model_image).all()
            assert (fit.model_images_of_planes[1] == fit.inversion_model_image).all()

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            galaxy_light = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))
            galaxy_pix = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=li_manual.grids, borders=li_manual.borders)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_manual, tracer=tracer)

            image_im = tracer._image_plane_image
            blurring_im = tracer._image_plane_blurring_image
            profile_model_image = li_manual.convolver_image.convolve_image(image_im, blurring_im)
            profile_subtracted_image = li_manual[:] - profile_model_image

            assert li_manual.grids.image.scaled_array_from_array_1d(profile_model_image) == \
                   pytest.approx(fit.profile_model_image, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(profile_subtracted_image) == \
                   pytest.approx(fit.profile_subtracted_image, 1e-4)

            mapper = pix.mapper_from_grids_and_borders(li_manual.grids, li_manual.borders)
            inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(mapper=mapper, regularization=reg,
                                                                                          image=profile_subtracted_image,
                                                                                          noise_map=li_manual.noise_map,
                                                                                          convolver=li_manual.convolver_mapping_matrix)

            model_image = profile_model_image + inversion.reconstructed_data_vector
            residuals = fitting.residuals_from_image_and_model(li_manual, model_image)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_manual.noise_map)

            assert li_manual.grids.image.scaled_array_from_array_1d(li_manual.noise_map) == pytest.approx(fit.noise_map, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(inversion.reconstructed_data_vector) == \
                   pytest.approx(fit.inversion_model_image, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(model_image) == pytest.approx(fit.model_image, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(residuals) == pytest.approx(fit.residuals, 1e-4)
            assert li_manual.grids.image.scaled_array_from_array_1d(chi_squareds) == pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(li_manual.noise_map)
            evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                                          inversion.log_det_curvature_reg_matrix_term,
                                                                          inversion.log_det_regularization_matrix_term,
                                                                          noise_term)

            assert evidence == fit.evidence

            fast_evidence = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                          tracer=tracer)
            assert fast_evidence == evidence


class TestHyperLensingProfileInversionFit:

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            hyp_galaxy = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)
            inv_galaxy = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[hyp_galaxy, hyp_galaxy],
                                                         source_galaxies=[inv_galaxy],
                                                         image_plane_grids=li_hyper_manual.grids,
                                                         borders=li_hyper_manual.borders)

            fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_hyper_manual, tracer=tracer)

            image_im = tracer._image_plane_image
            blurring_im = tracer._image_plane_blurring_image
            profile_model_image = li_hyper_manual.convolver_image.convolve_image(image_im, blurring_im)
            profile_subtracted_image = li_hyper_manual[:] - profile_model_image

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(profile_model_image) == \
                   pytest.approx(fit.profile_model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(profile_subtracted_image) == \
                   pytest.approx(fit.profile_subtracted_image, 1e-4)

            mapper = pix.mapper_from_grids_and_borders(li_hyper_manual.grids, li_hyper_manual.borders)
            inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(mapper=mapper, regularization=reg,
                        image=profile_subtracted_image, noise_map=li_hyper_manual.noise_map,
                        convolver=li_hyper_manual.convolver_mapping_matrix)

            model_image = profile_model_image + inversion.reconstructed_data_vector
            residuals = fitting.residuals_from_image_and_model(li_hyper_manual, model_image)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li_hyper_manual.noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(li_hyper_manual.noise_map) ==\
                   pytest.approx(fit.noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(inversion.reconstructed_data_vector) == \
                   pytest.approx(fit.inversion_model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(model_image) ==\
                   pytest.approx(fit.model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(chi_squareds) ==\
                   pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(li_hyper_manual.noise_map)
            evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                                  inversion.log_det_curvature_reg_matrix_term,
                                                                  inversion.log_det_regularization_matrix_term,
                                                                  noise_term)

            assert evidence == fit.evidence

            contributions = fitting.contributions_from_hyper_images_and_galaxies(li_hyper_manual.hyper_model_image,
                            li_hyper_manual.hyper_galaxy_images, [hyper_galaxy,hyper_galaxy],
                            li_hyper_manual.hyper_minimum_values)
            scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                               [hyper_galaxy, hyper_galaxy], li_hyper_manual.noise_map)

            scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
                mapper=mapper, regularization=reg, image=profile_subtracted_image, noise_map=scaled_noise_map,
                convolver=li_hyper_manual.convolver_mapping_matrix)

            scaled_model_image = profile_model_image + scaled_inversion.reconstructed_data_vector
            scaled_residuals = fitting.residuals_from_image_and_model(li_hyper_manual, scaled_model_image)
            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(scaled_residuals, scaled_noise_map)

            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[0]) == \
                   pytest.approx(fit.contributions[0], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(contributions[1]) == \
                   pytest.approx(fit.contributions[1], 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_noise_map) == \
                   pytest.approx(fit.scaled_noise_map, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_model_image) == \
                   pytest.approx(fit.scaled_model_image, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_residuals) == \
                   pytest.approx(fit.scaled_residuals, 1e-4)
            assert li_hyper_manual.grids.image.scaled_array_from_array_1d(scaled_chi_squareds) == \
                   pytest.approx(fit.scaled_chi_squareds, 1e-4)

            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_noise_map(scaled_noise_map)
            scaled_evidence = fitting.evidence_from_reconstruction_terms(
                scaled_chi_squared_term, scaled_inversion.regularization_term,
                scaled_inversion.log_det_curvature_reg_matrix_term, scaled_inversion.log_det_regularization_matrix_term,
                scaled_noise_term)

            assert scaled_evidence == fit.scaled_evidence

            fast_scaled_evidence = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(
                lensing_image=li_hyper_manual, tracer=tracer)

            assert fast_scaled_evidence == scaled_evidence


class MockTracerPositions:

    def __init__(self, positions, noise=None):
        self.positions = positions
        self.noise = noise


class TestPositionFit:

    def test__x1_positions__mock_position_tracer__maximum_separation_is_correct(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(2)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 3.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(1.0) + np.square(3.0))

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [-9.0, -4.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_positions__mock_position_tracer__maximum_separation_is_correct(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0], [0.1, 0.1], [-0.1, -0.1],
                                                          [0.3, 0.4], [-0.6, 0.5]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [8.0, 4.0], [-9.0, -4.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_sets_of_positions__multiple_sets_of_max_distances(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)

        assert fit.maximum_separations[0] == 1.0
        assert fit.maximum_separations[1] == np.sqrt(18)
        assert fit.maximum_separations[2] == np.sqrt(18)

    def test__likelihood__is_sum_of_separations_divided_by_noise(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)
        assert fit.chi_squareds[0] == 1.0
        assert fit.chi_squareds[1] == pytest.approx(18.0, 1e-4)
        assert fit.chi_squareds[2] == pytest.approx(18.0, 1e-4)
        assert fit.likelihood == pytest.approx(-0.5 * (1.0 + 18 + 18), 1e-4)

        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=2.0)
        assert fit.chi_squareds[0] == (1.0 / 2.0) ** 2.0
        assert fit.chi_squareds[1] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fit.chi_squareds[2] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fit.likelihood == pytest.approx(-0.5 * ((1.0 / 2.0) ** 2.0 + (np.sqrt(18.0) / 2.0) ** 2.0 +
                                                       (np.sqrt(18.0) / 2.0) ** 2.0), 1e-4)

    def test__threshold__if_not_met_returns_ray_tracing_exception(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = lensing_fitting.PositionFit(positions=tracer.positions, noise=1.0)

        assert fit.maximum_separation_within_threshold(threshold=100.0) == True
        assert fit.maximum_separation_within_threshold(threshold=0.1) == False
