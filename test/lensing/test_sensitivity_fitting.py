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
from autolens.lensing import sensitivity_fitting
from autolens.lensing import plane as pl
from autolens.lensing import ray_tracing
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from test.mock.mock_profiles import MockLightProfile
from test.mock.mock_galaxy import MockHyperGalaxy
from test.mock.mock_lensing import MockTracer


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
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)), effective_exposure_map=3.0*np.ones((4,4)),
                     background_sky_map=4.0*np.ones((4,4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)


class TestMockImageWithSubhalo:

    def test__mock_tracer__convolver_blur__mock_image_is_tracer_image(self, li_blur):

        tracer_sensitivity = MockTracer(images=[li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image)],
                            blurring_images=[np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])],
                            has_light_profile=True, has_hyper_galaxy=False, has_pixelization=False)

        sensitivity_images = sensitivity_fitting.sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=[li_blur], tracer_sensitive=tracer_sensitivity, add_noise=False)

        assert (sensitivity_images[0] == np.array([9.0, 9.0, 9.0, 9.0])).all()

        assert (sensitivity_images[0].image == np.array([[0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 9.0, 9.0, 0.0],
                                                  [0.0, 9.0, 9.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0]])).all()

        assert (sensitivity_images[0].image.noise_map == li_blur.noise_map).all()
        assert (sensitivity_images[0].noise_map == np.ones(4)).all()
        assert (sensitivity_images[0].psf == li_blur.psf).all()

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

        tracer_sensitivity = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.grids])

        sensitivity_images = sensitivity_fitting.sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=[li], tracer_sensitive=tracer_sensitivity, add_noise=False)

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        central_values = tracer_sensitivity._image_plane_images[0]
        blurring_values = tracer_sensitivity._image_plane_blurring_images[0]

        tracer_blurred_image_manual_0 = 2.0 * central_values[0] + 3.0 * central_values[2] + blurring_values[4]
        tracer_blurred_image_manual_1 = 2.0 * central_values[1] + 3.0 * central_values[3] + central_values[0]
        tracer_blurred_image_manual_2 = 2.0 * central_values[2] + 3.0 * blurring_values[9] + blurring_values[6]
        tracer_blurred_image_manual_3 = 2.0 * central_values[3] + 3.0 * blurring_values[10] + central_values[2]

        assert tracer_blurred_image_manual_0 == pytest.approx(sensitivity_images[0][0], 1e-6)
        assert tracer_blurred_image_manual_1 == pytest.approx(sensitivity_images[0][1], 1e-6)
        assert tracer_blurred_image_manual_2 == pytest.approx(sensitivity_images[0][2], 1e-6)
        assert tracer_blurred_image_manual_3 == pytest.approx(sensitivity_images[0][3], 1e-6)

        assert tracer_blurred_image_manual_0 == pytest.approx(sensitivity_images[0].image[1,1], 1e-6)
        assert tracer_blurred_image_manual_1 == pytest.approx(sensitivity_images[0].image[1,2], 1e-6)
        assert tracer_blurred_image_manual_2 == pytest.approx(sensitivity_images[0].image[2,1], 1e-6)
        assert tracer_blurred_image_manual_3 == pytest.approx(sensitivity_images[0].image[2,2], 1e-6)


class TestAddPoissonNoise:

    def test__mock_arrays_all_1s__poisson_noise_is_added_correct(self, li_blur):

        _mock_array = np.ones(4)
        _mock_array_with_sky = _mock_array + 4.0*np.ones(4)
        _mock_array_with_sky_and_noise = _mock_array_with_sky + image.generate_poisson_noise(image=_mock_array_with_sky,
                                                                 effective_exposure_map=3.0*np.ones(4), seed=1)

        _mock_array_with_noise = _mock_array_with_sky_and_noise - 4.0*np.ones(4)

        _mock_arrays_with_noise = sensitivity_fitting.add_poisson_noise_to_mock_arrays(_mock_arrays=[_mock_array],
                                                                                       lensing_images=[li_blur],
                                                                                       seed=1)
        assert (_mock_array_with_noise == _mock_arrays_with_noise[0]).all()


class TestSensitivityProfileFit:

    def test__tracer_and_tracer_sensitivity_are_identical__no_noise_added__likelihood_is_noise_term(self, li_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grids=[li_blur.grids])

        fit = sensitivity_fitting.SensitivityProfileFit(lensing_images=[li_blur], tracer_normal=tracer,
                                                        tracer_sensitive=tracer, add_noise=False, noise_seed=1)

        sensitivity_images = sensitivity_fitting.sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=[li_blur], tracer_sensitive=tracer, add_noise=False, noise_seed=1)

        assert (fit.sensitivity_images[0] == sensitivity_images[0]).all()

        assert (fit.fit_normal._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_normal._noise_maps[0] == sensitivity_images[0].noise_map).all()
        assert (fit.fit_normal._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_normal._residuals[0] == np.zeros(4)).all()
        assert (fit.fit_normal._chi_squareds[0] == np.zeros(4)).all()

        assert (fit.fit_sensitive._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_sensitive._noise_maps[0] == sensitivity_images[0].noise_map).all()
        assert (fit.fit_sensitive._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_sensitive._residuals[0] == np.zeros(4)).all()
        assert (fit.fit_sensitive._chi_squareds[0] == np.zeros(4)).all()

        noise_term = sum(fitting.noise_terms_from_noise_maps(noise_maps=[sensitivity_images[0].noise_map]))
        assert fit.fit_normal.likelihood == -0.5 * noise_term
        assert fit.fit_sensitive.likelihood == -0.5 * noise_term

        assert fit.likelihood == 0.0

        fast_likelihood = sensitivity_fitting.SensitivityProfileFit.fast_likelihood(lensing_images=[li_blur],
                                                                                    tracer_normal=tracer,
                                                                                    tracer_sensitive=tracer,
                                                                                    add_noise=False, noise_seed=1)

        assert fit.likelihood == fast_likelihood

    def test__tracer_and_tracer_sensitivity_are_identical__noise_added__likelihood_is_noise_term(self, li_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grids=[li_blur.grids])

        fit = sensitivity_fitting.SensitivityProfileFit(lensing_images=[li_blur], tracer_normal=tracer,
                                                        tracer_sensitive=tracer, add_noise=True, noise_seed=1)

        sensitivity_images = sensitivity_fitting.sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=[li_blur], tracer_sensitive=tracer, add_noise=True, noise_seed=1)

        assert (fit.sensitivity_images[0] == sensitivity_images[0]).all()

        assert (fit.fit_normal._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_normal._noise_maps[0] == sensitivity_images[0].noise_map).all()
        assert (fit.fit_normal._datas[0] == sensitivity_images[0]).all()

        _model_datas = fitting.blur_images_including_blurring_regions(images=tracer._image_plane_images,
                                                                    blurring_images=tracer._image_plane_blurring_images,
                                                                    convolvers=[li_blur.convolver_image])

        assert (fit.fit_normal._model_datas[0] == _model_datas[0]).all()

        _residuals = fitting.residuals_from_datas_and_model_datas(datas=sensitivity_images,
                                                                  model_datas=_model_datas)

        assert (fit.fit_normal._residuals[0] == _residuals[0]).all()

        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals=_residuals,
                                                                           noise_maps=[li_blur.noise_map])

        assert (fit.fit_normal._chi_squareds[0] == _chi_squareds).all()


        assert (fit.fit_sensitive._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_sensitive._noise_maps[0] == sensitivity_images[0].noise_map).all()

        _model_datas = fitting.blur_images_including_blurring_regions(images=tracer._image_plane_images,
                                                                    blurring_images=tracer._image_plane_blurring_images,
                                                                    convolvers=[li_blur.convolver_image])

        assert (fit.fit_sensitive._model_datas[0] == _model_datas[0]).all()

        _residuals = fitting.residuals_from_datas_and_model_datas(datas=sensitivity_images,
                                                                  model_datas=_model_datas)

        assert (fit.fit_sensitive._residuals[0] == _residuals[0]).all()

        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals=_residuals,
                                                                           noise_maps=[li_blur.noise_map])

        assert (fit.fit_sensitive._chi_squareds[0] == _chi_squareds).all()

        chi_squared_term = sum(fitting.chi_squared_terms_from_chi_squareds(chi_squareds=_chi_squareds))
        noise_term = sum(fitting.noise_terms_from_noise_maps(noise_maps=[sensitivity_images[0].noise_map]))
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared_term + noise_term)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared_term + noise_term)

        assert fit.likelihood == 0.0

        fast_likelihood = sensitivity_fitting.SensitivityProfileFit.fast_likelihood(lensing_images=[li_blur],
                                                                                    tracer_normal=tracer,
                                                                                    tracer_sensitive=tracer, add_noise=True, noise_seed=1)

        assert fit.likelihood == fast_likelihood

    def test__tracers_are_different__likelihood_is_non_zero(self, li_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grids=[li_blur.grids])

        tracer_sensitivity = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
                                                     image_plane_grids=[li_blur.grids])

        fit = sensitivity_fitting.SensitivityProfileFit(lensing_images=[li_blur], tracer_normal=tracer,
                                                        tracer_sensitive=tracer_sensitivity, add_noise=False)

        sensitivity_images = sensitivity_fitting.sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=[li_blur], tracer_sensitive=tracer_sensitivity, add_noise=False)

        assert (fit.sensitivity_images[0] == sensitivity_images[0]).all()

        assert (fit.fit_normal._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_normal._noise_maps[0] == sensitivity_images[0].noise_map).all()

        _model_datas = fitting.blur_images_including_blurring_regions(images=tracer._image_plane_images,
                                                                    blurring_images=tracer._image_plane_blurring_images,
                                                                    convolvers=[li_blur.convolver_image])

        assert (fit.fit_normal._model_datas[0] == _model_datas[0]).all()

        _residuals = fitting.residuals_from_datas_and_model_datas(datas=sensitivity_images,
                                                                  model_datas=_model_datas)

        assert (fit.fit_normal._residuals[0] == _residuals[0]).all()

        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals=_residuals,
                                                                           noise_maps=[li_blur.noise_map])

        assert (fit.fit_normal._chi_squareds[0] == _chi_squareds).all()

        assert (fit.fit_sensitive._datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_sensitive._noise_maps[0] == sensitivity_images[0].noise_map).all()
        assert (fit.fit_sensitive._model_datas[0] == sensitivity_images[0]).all()
        assert (fit.fit_sensitive._residuals[0] == np.zeros(4)).all()
        assert (fit.fit_sensitive._chi_squareds[0] == np.zeros(4)).all()

        chi_squared_term = sum(fitting.chi_squared_terms_from_chi_squareds(chi_squareds=_chi_squareds))
        noise_term = sum(fitting.noise_terms_from_noise_maps(noise_maps=[sensitivity_images[0].noise_map]))
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared_term + noise_term)
        assert fit.fit_sensitive.likelihood == -0.5 * noise_term

        assert fit.likelihood == 0.5 * chi_squared_term

        fast_likelihood = sensitivity_fitting.SensitivityProfileFit.fast_likelihood(lensing_images=[li_blur],
                                                                                    tracer_normal=tracer,
                                                                                    tracer_sensitive=tracer_sensitivity, add_noise=False)

        assert fit.likelihood == fast_likelihood