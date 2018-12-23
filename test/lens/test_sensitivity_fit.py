import numpy as np
import pytest

from autofit.core import fitting_util
from autolens.data.imaging import ccd
from autolens.data.array import mask as mask
from autolens.model.galaxy import galaxy as g
from autolens.lens.util import lens_fit_util as util
from autolens.lens import lens_image
from autolens.lens import sensitivity_fit
from autolens.lens import ray_tracing
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp

@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return g.Galaxy(light_profile=sersic)

@pytest.fixture(name='si_blur')
def make_si_blur():
    psf = ccd.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    im = ccd.CCD(image=5.0 * np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)),
                 exposure_time_map=3.0 * np.ones((4, 4)), background_sky_map=4.0 * np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lens_image.LensImage(image=im, mask=ma, sub_grid_size=2)


class TestSensitivityProfileFit:

    def test__tracer_and_tracer_sensitive_are_identical__added__likelihood_is_noise_term(self, si_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stack=si_blur.grid_stack)

        fit = sensitivity_fit.SensitivityProfileFit(lens_image=si_blur, tracer_normal=tracer,
                                                    tracer_sensitive=tracer)

        assert (fit.fit_normal.image == si_blur.image).all()
        assert (fit.fit_normal.noise_map == si_blur.noise_map).all()

        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=si_blur.convolver_image)

        model_image = si_blur.map_to_scaled_array(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image == model_image).all()

        residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=si_blur.image, mask=si_blur.mask,
                                                                             model_data=model_image)
        assert (fit.fit_normal.residual_map == residual_map).all()

        chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=si_blur.mask, noise_map=si_blur.noise_map)

        assert (fit.fit_normal.chi_squared_map == chi_squared_map).all()

        assert (fit.fit_sensitive.image == si_blur.image).all()
        assert (fit.fit_sensitive.noise_map == si_blur.noise_map).all()
        assert (fit.fit_sensitive.model_image == model_image).all()
        assert (fit.fit_sensitive.residual_map == residual_map).all()
        assert (fit.fit_sensitive.chi_squared_map == chi_squared_map).all()

        chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map, 
                                                                             mask=si_blur.mask)
        noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=si_blur.mask,
                                                                                       noise_map=si_blur.noise_map)
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared + noise_normalization)

        assert fit.figure_of_merit == 0.0

    def test__tracers_are_different__likelihood_is_non_zero(self, si_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                            image_plane_grid_stack=si_blur.grid_stack)

        tracer_sensitive = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
                                                               image_plane_grid_stack=si_blur.grid_stack)
        fit = sensitivity_fit.SensitivityProfileFit(lens_image=si_blur, tracer_normal=tracer_normal,
                                                    tracer_sensitive=tracer_sensitive)

        assert (fit.fit_normal.image == si_blur.image).all()
        assert (fit.fit_normal.noise_map == si_blur.noise_map).all()

        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer_normal.image_plane_image_1d, blurring_image_1d=tracer_normal.image_plane_blurring_image_1d,
            convolver=si_blur.convolver_image)

        model_image = si_blur.map_to_scaled_array(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image == model_image).all()

        residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=si_blur.image, mask=si_blur.mask,
                                                                             model_data=model_image)
        assert (fit.fit_normal.residual_map == residual_map).all()

        chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=si_blur.mask, noise_map=si_blur.noise_map)

        assert (fit.fit_normal.chi_squared_map == chi_squared_map).all()


        assert (fit.fit_sensitive.image == si_blur.image).all()
        assert (fit.fit_sensitive.noise_map == si_blur.noise_map).all()
        
        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer_sensitive.image_plane_image_1d,
            blurring_image_1d=tracer_sensitive.image_plane_blurring_image_1d,
            convolver=si_blur.convolver_image)

        model_image = si_blur.map_to_scaled_array(array_1d=model_image_1d)
        
        assert (fit.fit_sensitive.model_image == model_image).all()
        
        residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=si_blur.image, mask=si_blur.mask,
                                                                             model_data=model_image)
        
        assert (fit.fit_sensitive.residual_map == residual_map).all()
        
        chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=si_blur.mask, noise_map=si_blur.noise_map)
        
        assert (fit.fit_sensitive.chi_squared_map == chi_squared_map).all()

        chi_squared_normal = fitting_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_normal.chi_squared_map, mask=si_blur.mask)
        chi_squared_sensitive = fitting_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_sensitive.chi_squared_map, mask=si_blur.mask)
        noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=si_blur.mask,
                                                                                       noise_map=si_blur.noise_map)
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared_normal + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared_sensitive + noise_normalization)

        assert fit.figure_of_merit == fit.fit_sensitive.likelihood - fit.fit_normal.likelihood