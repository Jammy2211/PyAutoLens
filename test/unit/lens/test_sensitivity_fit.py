import numpy as np
import pytest

from autofit.tools import fit_util
from autolens.data import ccd
from autolens.data.array import mask as mask
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.inversion import inversions as inv
from autolens.lens.util import lens_fit_util as util
from autolens.lens import lens_data as ld
from autolens.lens import sensitivity_fit
from autolens.lens import ray_tracing
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp

@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return g.Galaxy(light_profile=sersic)

@pytest.fixture(name='lens_data_blur')
def make_lens_data_blur():

    psf = ccd.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    im = ccd.CCDData(image=5.0 * np.ones((6, 6)), pixel_scale=1.0, psf=psf, noise_map=np.ones((6, 6)),
                     exposure_time_map=3.0 * np.ones((6, 6)), background_sky_map=4.0 * np.ones((6, 6)))

    ma = np.array([[True,  True,  True,  True,  True, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True,  True,  True,  True,  True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return ld.LensData(ccd_data=im, mask=ma, sub_grid_size=2)


class TestSensitivityProfileFit:

    def test__tracer_and_tracer_sensitive_are_identical__added__likelihood_is_noise_term(self, lens_data_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stack=lens_data_blur.grid_stack)

        fit = sensitivity_fit.SensitivityProfileFit(lens_data=lens_data_blur, tracer_normal=tracer,
                                                    tracer_sensitive=tracer)

        assert (fit.fit_normal.image == lens_data_blur.image).all()
        assert (fit.fit_normal.noise_map == lens_data_blur.noise_map).all()

        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lens_data_blur.convolver_image)

        model_image = lens_data_blur.map_to_scaled_array(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image == model_image).all()

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image, mask=lens_data_blur.mask,
                                                                             model_data=model_image)
        assert (fit.fit_normal.residual_map == residual_map).all()

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)

        assert (fit.fit_normal.chi_squared_map == chi_squared_map).all()

        assert (fit.fit_sensitive.image == lens_data_blur.image).all()
        assert (fit.fit_sensitive.noise_map == lens_data_blur.noise_map).all()
        assert (fit.fit_sensitive.model_image == model_image).all()
        assert (fit.fit_sensitive.residual_map == residual_map).all()
        assert (fit.fit_sensitive.chi_squared_map == chi_squared_map).all()

        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map, 
                                                                             mask=lens_data_blur.mask)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=lens_data_blur.mask,
                                                                                       noise_map=lens_data_blur.noise_map)
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared + noise_normalization)

        assert fit.figure_of_merit == 0.0

    def test__tracers_are_different__likelihood_is_non_zero(self, lens_data_blur):

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                            image_plane_grid_stack=lens_data_blur.grid_stack)

        tracer_sensitive = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
                                                               image_plane_grid_stack=lens_data_blur.grid_stack)

        fit = sensitivity_fit.SensitivityProfileFit(lens_data=lens_data_blur, tracer_normal=tracer_normal,
                                                    tracer_sensitive=tracer_sensitive)

        assert (fit.fit_normal.image == lens_data_blur.image).all()
        assert (fit.fit_normal.noise_map == lens_data_blur.noise_map).all()

        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer_normal.image_plane_image_1d, blurring_image_1d=tracer_normal.image_plane_blurring_image_1d,
            convolver=lens_data_blur.convolver_image)

        model_image = lens_data_blur.map_to_scaled_array(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image == model_image).all()

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image, mask=lens_data_blur.mask,
                                                                             model_data=model_image)
        assert (fit.fit_normal.residual_map == residual_map).all()

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)

        assert (fit.fit_normal.chi_squared_map == chi_squared_map).all()


        assert (fit.fit_sensitive.image == lens_data_blur.image).all()
        assert (fit.fit_sensitive.noise_map == lens_data_blur.noise_map).all()
        
        model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer_sensitive.image_plane_image_1d,
            blurring_image_1d=tracer_sensitive.image_plane_blurring_image_1d,
            convolver=lens_data_blur.convolver_image)

        model_image = lens_data_blur.map_to_scaled_array(array_1d=model_image_1d)
        
        assert (fit.fit_sensitive.model_image == model_image).all()
        
        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image, mask=lens_data_blur.mask,
                                                                             model_data=model_image)
        
        assert (fit.fit_sensitive.residual_map == residual_map).all()
        
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)
        
        assert (fit.fit_sensitive.chi_squared_map == chi_squared_map).all()

        chi_squared_normal = fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_normal.chi_squared_map, mask=lens_data_blur.mask)
        chi_squared_sensitive = fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_sensitive.chi_squared_map, mask=lens_data_blur.mask)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=lens_data_blur.mask,
                                                                                       noise_map=lens_data_blur.noise_map)
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared_normal + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared_sensitive + noise_normalization)

        assert fit.figure_of_merit == fit.fit_sensitive.likelihood - fit.fit_normal.likelihood

        fit_from_factory = sensitivity_fit.fit_lens_data_with_sensitivity_tracers(lens_data=lens_data_blur,
                                                                                 tracer_normal=tracer_normal,
                                                                                 tracer_sensitive=tracer_sensitive)

        assert fit.figure_of_merit == fit_from_factory.figure_of_merit


class TestSensitivityInversionFit:

    def test__tracer_and_tracer_sensitive_are_identical__added__likelihood_is_noise_term(self, lens_data_blur):

        pixelization = pix.Rectangular(shape=(3, 3))
        regularization = reg.Constant(coefficients=(1.0,))

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(pixelization=pixelization, regularization=regularization)

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stack=lens_data_blur.grid_stack)

        fit = sensitivity_fit.SensitivityInversionFit(lens_data=lens_data_blur, tracer_normal=tracer,
                                                      tracer_sensitive=tracer)

        assert (fit.fit_normal.image == lens_data_blur.image).all()
        assert (fit.fit_normal.noise_map == lens_data_blur.noise_map).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(grid_stack=tracer.source_plane.grid_stack, border=None)
        inversion = inv.inversion_from_image_mapper_and_regularization(mapper=mapper,
                                                                      regularization=regularization,
                                                                      image_1d=lens_data_blur.image_1d,
                                                                      noise_map_1d=lens_data_blur.noise_map_1d,
                                                                      convolver=lens_data_blur.convolver_mapping_matrix)

        assert fit.fit_normal.model_image == pytest.approx(inversion.reconstructed_data, 1.0e-4)

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image,
                                                                           mask=lens_data_blur.mask,
                                                                           model_data=inversion.reconstructed_data)

        assert fit.fit_normal.residual_map == pytest.approx(residual_map, 1.0e-4)

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)

        assert fit.fit_normal.chi_squared_map == pytest.approx(chi_squared_map, 1.0e-4)

        assert fit.fit_sensitive.image == pytest.approx(lens_data_blur.image, 1.0e-4)
        assert fit.fit_sensitive.noise_map == pytest.approx(lens_data_blur.noise_map, 1.0e-4)
        assert fit.fit_sensitive.model_image == pytest.approx(inversion.reconstructed_data, 1.0e-4)
        assert fit.fit_sensitive.residual_map == pytest.approx(residual_map, 1.0e-4)
        assert fit.fit_sensitive.chi_squared_map == pytest.approx(chi_squared_map, 1.0e-4)

        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                         mask=lens_data_blur.mask)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=lens_data_blur.mask,
                                                                                   noise_map=lens_data_blur.noise_map)

        assert fit.fit_normal.likelihood == -0.5 * (chi_squared + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared + noise_normalization)

        assert fit.figure_of_merit == 0.0

    def test__tracers_are_different__likelihood_is_non_zero(self, lens_data_blur):

        pixelization = pix.Rectangular(shape=(3, 3))
        regularization = reg.Constant(coefficients=(1.0,))

        g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(pixelization=pixelization, regularization=regularization)

        tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                            image_plane_grid_stack=lens_data_blur.grid_stack)

        tracer_sensitive = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
                                                               image_plane_grid_stack=lens_data_blur.grid_stack)

        fit = sensitivity_fit.SensitivityInversionFit(lens_data=lens_data_blur, tracer_normal=tracer_normal,
                                                      tracer_sensitive=tracer_sensitive)

        assert (fit.fit_normal.image == lens_data_blur.image).all()
        assert (fit.fit_normal.noise_map == lens_data_blur.noise_map).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(grid_stack=tracer_normal.source_plane.grid_stack,
                                                                border=None)
        inversion = inv.inversion_from_image_mapper_and_regularization(mapper=mapper,
                                                                      regularization=regularization,
                                                                      image_1d=lens_data_blur.image_1d,
                                                                      noise_map_1d=lens_data_blur.noise_map_1d,
                                                                      convolver=lens_data_blur.convolver_mapping_matrix)

        assert fit.fit_normal.model_image == pytest.approx(inversion.reconstructed_data, 1.0e-4)

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image,
                                                                           mask=lens_data_blur.mask,
                                                                           model_data=inversion.reconstructed_data)

        assert fit.fit_normal.residual_map == pytest.approx(residual_map, 1.0e-4)

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)

        assert fit.fit_normal.chi_squared_map == pytest.approx(chi_squared_map, 1.0e-4)
        
        assert (fit.fit_sensitive.image == lens_data_blur.image).all()
        assert (fit.fit_sensitive.noise_map == lens_data_blur.noise_map).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(grid_stack=tracer_sensitive.source_plane.grid_stack,
                                                                border=None)
        inversion = inv.inversion_from_image_mapper_and_regularization(mapper=mapper,
                                                                      regularization=regularization,
                                                                      image_1d=lens_data_blur.image_1d,
                                                                      noise_map_1d=lens_data_blur.noise_map_1d,
                                                                      convolver=lens_data_blur.convolver_mapping_matrix)

        assert fit.fit_sensitive.model_image == pytest.approx(inversion.reconstructed_data, 1.0e-4)

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=lens_data_blur.image,
                                                                           mask=lens_data_blur.mask,
                                                                           model_data=inversion.reconstructed_data)

        assert fit.fit_sensitive.residual_map == pytest.approx(residual_map, 1.0e-4)

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                        mask=lens_data_blur.mask, noise_map=lens_data_blur.noise_map)

        assert fit.fit_sensitive.chi_squared_map == pytest.approx(chi_squared_map, 1.0e-4)

        chi_squared_normal = fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_normal.chi_squared_map, mask=lens_data_blur.mask)
        chi_squared_sensitive = fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_sensitive.chi_squared_map, mask=lens_data_blur.mask)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=lens_data_blur.mask,
                                                                                   noise_map=lens_data_blur.noise_map)
        assert fit.fit_normal.likelihood == -0.5 * (chi_squared_normal + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (chi_squared_sensitive + noise_normalization)

        assert fit.figure_of_merit == fit.fit_sensitive.likelihood - fit.fit_normal.likelihood

        fit_from_factory = sensitivity_fit.fit_lens_data_with_sensitivity_tracers(lens_data=lens_data_blur,
                                                                                 tracer_normal=tracer_normal,
                                                                                 tracer_sensitive=tracer_sensitive)

        assert fit.figure_of_merit == fit_from_factory.figure_of_merit