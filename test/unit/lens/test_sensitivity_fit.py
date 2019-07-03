import numpy as np
import pytest

import autofit as af
from autolens.data import ccd
from autolens.data.array import mask as mask
from autolens.lens import lens_data as ld
from autolens.lens import ray_tracing
from autolens.lens import sensitivity_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import inversions as inv
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                               effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return g.Galaxy(redshift=0.5, light_profile=sersic)


@pytest.fixture(name='ld_blur')
def make_ld_blur():
    psf = ccd.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]])), pixel_scale=1.0,
                  renormalize=False)
    im = ccd.CCDData(image=5.0 * np.ones((6, 6)), pixel_scale=1.0, psf=psf,
                     noise_map=np.ones((6, 6)),
                     exposure_time_map=3.0 * np.ones((6, 6)),
                     background_sky_map=4.0 * np.ones((6, 6)))

    ma = np.array([[True, True, True, True, True, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True, False, False, False, False, True],
                   [True, True, True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return ld.LensData(ccd_data=im, mask=ma, sub_grid_size=2)


class TestSensitivityProfileFit:

    def test__tracer_and_tracer_sensitive_are_identical__added__likelihood_is_noise_term(
            self, ld_blur):
        g0 = g.Galaxy(redshift=0.5,
                      mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0],
                                                     source_galaxies=[g1],
                                                     image_plane_grid_stack=ld_blur.grid_stack)

        fit = sensitivity_fit.SensitivityProfileFit(lens_data=ld_blur,
                                                    tracer_normal=tracer,
                                                    tracer_sensitive=tracer)

        assert (fit.fit_normal.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_normal.noise_map_2d == ld_blur.noise_map_2d).all()

        model_image_1d = tracer.blurred_profile_image_plane_image_1d_from_convolver_image(
            convolver_image=ld_blur.convolver_image)

        model_image_2d = ld_blur.scaled_array_2d_from_array_1d(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image_2d == model_image_2d).all()

        residual_map_1d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_1d, mask=ld_blur.mask_1d, model_data=model_image_1d)

        residual_map_2d = ld_blur.scaled_array_2d_from_array_1d(array_1d=residual_map_1d)
        assert (fit.fit_normal.residual_map_2d == residual_map_2d).all()

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=ld_blur.mask_2d, noise_map=ld_blur.noise_map_2d)

        assert (fit.fit_normal.chi_squared_map_2d == chi_squared_map_2d).all()

        assert (fit.fit_sensitive.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_sensitive.noise_map_2d == ld_blur.noise_map_2d).all()
        assert (fit.fit_sensitive.model_image_2d == model_image_2d).all()
        assert (fit.fit_sensitive.residual_map_2d == residual_map_2d).all()
        assert (fit.fit_sensitive.chi_squared_map_2d == chi_squared_map_2d).all()

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=ld_blur.mask_2d)
        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=ld_blur.mask_2d, noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.likelihood == -0.5 * (chi_squared + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (
                chi_squared + noise_normalization)

        assert fit.figure_of_merit == 0.0

    def test__tracers_are_different__likelihood_is_non_zero(self, ld_blur):
        g0 = g.Galaxy(redshift=0.5,
                      mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(redshift=0.5,
                              subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0],
                                                            source_galaxies=[g1],
                                                            image_plane_grid_stack=ld_blur.grid_stack)

        tracer_sensitive = ray_tracing.TracerImageSourcePlanes(
            lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
            image_plane_grid_stack=ld_blur.grid_stack)

        fit = sensitivity_fit.SensitivityProfileFit(lens_data=ld_blur,
                                                    tracer_normal=tracer_normal,
                                                    tracer_sensitive=tracer_sensitive)

        assert (fit.fit_normal.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_normal.noise_map_2d == ld_blur.noise_map_2d).all()

        model_image_1d = tracer_normal.blurred_profile_image_plane_image_1d_from_convolver_image(
            convolver_image=ld_blur.convolver_image)

        model_image_2d = ld_blur.scaled_array_2d_from_array_1d(array_1d=model_image_1d)

        assert (fit.fit_normal.model_image_2d == model_image_2d).all()

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_2d, mask=ld_blur.mask_2d, model_data=model_image_2d)

        assert (fit.fit_normal.residual_map_2d == residual_map_2d).all()

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d, mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert (fit.fit_normal.chi_squared_map_2d == chi_squared_map_2d).all()

        assert (fit.fit_sensitive.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_sensitive.noise_map_2d == ld_blur.noise_map_2d).all()

        model_image_1d = tracer_sensitive.blurred_profile_image_plane_image_1d_from_convolver_image(
            convolver_image=ld_blur.convolver_image)

        model_image_2d = ld_blur.scaled_array_2d_from_array_1d(array_1d=model_image_1d)

        assert (fit.fit_sensitive.model_image_2d == model_image_2d).all()

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_2d, mask=ld_blur.mask_2d, model_data=model_image_2d)

        assert (fit.fit_sensitive.residual_map_2d == residual_map_2d).all()

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d, mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert (fit.fit_sensitive.chi_squared_map_2d == chi_squared_map_2d).all()

        chi_squared_normal = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_normal.chi_squared_map_2d, mask=ld_blur.mask_2d)
        chi_squared_sensitive = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_sensitive.chi_squared_map_2d, mask=ld_blur.mask_2d)
        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=ld_blur.mask_2d, noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.likelihood == -0.5 * (
                chi_squared_normal + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (
                chi_squared_sensitive + noise_normalization)

        assert fit.figure_of_merit == fit.fit_sensitive.likelihood - fit.fit_normal.likelihood

        fit_from_factory = sensitivity_fit.fit_lens_data_with_sensitivity_tracers(
            lens_data=ld_blur,
            tracer_normal=tracer_normal,
            tracer_sensitive=tracer_sensitive)

        assert fit.figure_of_merit == fit_from_factory.figure_of_merit


class TestSensitivityInversionFit:

    def test__tracer_and_tracer_sensitive_are_identical__added__likelihood_is_noise_term(
            self, ld_blur):
        pixelization = pix.Rectangular(shape=(3, 3))
        regularization = reg.Constant(coefficients=(1.0,))

        g0 = g.Galaxy(redshift=0.5,
                      mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g1 = g.Galaxy(redshift=1.0, pixelization=pixelization,
                      regularization=regularization)

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0],
                                                     source_galaxies=[g1],
                                                     image_plane_grid_stack=ld_blur.grid_stack)

        fit = sensitivity_fit.SensitivityInversionFit(lens_data=ld_blur,
                                                      tracer_normal=tracer,
                                                      tracer_sensitive=tracer)

        assert (fit.fit_normal.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_normal.noise_map_2d == ld_blur.noise_map_2d).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(
            grid_stack=tracer.source_plane.grid_stack, border=None)
        inversion = inv.Inversion.from_data_1d_mapper_and_regularization(
            mapper=mapper, regularization=regularization, image_1d=ld_blur.image_1d,
            noise_map_1d=ld_blur.noise_map_1d,
            convolver=ld_blur.convolver_mapping_matrix)

        assert fit.fit_normal.model_image_2d == pytest.approx(
            inversion.reconstructed_data_2d, 1.0e-4)

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_2d, mask=ld_blur.mask_2d,
            model_data=inversion.reconstructed_data_2d)

        assert fit.fit_normal.residual_map_2d == pytest.approx(residual_map_2d, 1.0e-4)

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d, mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.chi_squared_map_2d == pytest.approx(chi_squared_map_2d,
                                                                  1.0e-4)

        assert fit.fit_sensitive.image_2d == pytest.approx(ld_blur.image_2d, 1.0e-4)
        assert fit.fit_sensitive.noise_map_2d == pytest.approx(ld_blur.noise_map_2d,
                                                               1.0e-4)
        assert fit.fit_sensitive.model_image_2d == pytest.approx(
            inversion.reconstructed_data_2d, 1.0e-4)
        assert fit.fit_sensitive.residual_map_2d == pytest.approx(residual_map_2d,
                                                                  1.0e-4)
        assert fit.fit_sensitive.chi_squared_map_2d == pytest.approx(chi_squared_map_2d,
                                                                     1.0e-4)

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d,
            mask=ld_blur.mask_2d)
        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.likelihood == -0.5 * (chi_squared + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (
                chi_squared + noise_normalization)

        assert fit.figure_of_merit == 0.0

    def test__tracers_are_different__likelihood_is_non_zero(self, ld_blur):
        pixelization = pix.Rectangular(shape=(3, 3))
        regularization = reg.Constant(coefficients=(1.0,))

        g0 = g.Galaxy(redshift=0.5,
                      mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
        g0_subhalo = g.Galaxy(redshift=0.5,
                              subhalo=mp.SphericalIsothermal(einstein_radius=0.1))
        g1 = g.Galaxy(redshift=1.0, pixelization=pixelization,
                      regularization=regularization)

        tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0],
                                                            source_galaxies=[g1],
                                                            image_plane_grid_stack=ld_blur.grid_stack)

        tracer_sensitive = ray_tracing.TracerImageSourcePlanes(
            lens_galaxies=[g0, g0_subhalo], source_galaxies=[g1],
            image_plane_grid_stack=ld_blur.grid_stack)

        fit = sensitivity_fit.SensitivityInversionFit(lens_data=ld_blur,
                                                      tracer_normal=tracer_normal,
                                                      tracer_sensitive=tracer_sensitive)

        assert (fit.fit_normal.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_normal.noise_map_2d == ld_blur.noise_map_2d).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(
            grid_stack=tracer_normal.source_plane.grid_stack,
            border=None)
        inversion = inv.Inversion.from_data_1d_mapper_and_regularization(
            mapper=mapper, regularization=regularization, image_1d=ld_blur.image_1d,
            noise_map_1d=ld_blur.noise_map_1d,
            convolver=ld_blur.convolver_mapping_matrix)

        assert fit.fit_normal.model_image_2d == pytest.approx(
            inversion.reconstructed_data_2d, 1.0e-4)

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_2d, mask=ld_blur.mask_2d,
            model_data=inversion.reconstructed_data_2d)

        assert fit.fit_normal.residual_map_2d == pytest.approx(residual_map_2d, 1.0e-4)

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d, mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.chi_squared_map_2d == pytest.approx(chi_squared_map_2d,
                                                                  1.0e-4)

        assert (fit.fit_sensitive.image_2d == ld_blur.image_2d).all()
        assert (fit.fit_sensitive.noise_map_2d == ld_blur.noise_map_2d).all()

        mapper = pixelization.mapper_from_grid_stack_and_border(
            grid_stack=tracer_sensitive.source_plane.grid_stack,
            border=None)
        inversion = inv.Inversion.from_data_1d_mapper_and_regularization(
            mapper=mapper, regularization=regularization, image_1d=ld_blur.image_1d,
            noise_map_1d=ld_blur.noise_map_1d,
            convolver=ld_blur.convolver_mapping_matrix)

        assert fit.fit_sensitive.model_image_2d == pytest.approx(
            inversion.reconstructed_data_2d, 1.0e-4)

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=ld_blur.image_2d, mask=ld_blur.mask_2d,
            model_data=inversion.reconstructed_data_2d)

        assert fit.fit_sensitive.residual_map_2d == pytest.approx(residual_map_2d,
                                                                  1.0e-4)

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d, mask=ld_blur.mask_2d,
            noise_map=ld_blur.noise_map_2d)

        assert fit.fit_sensitive.chi_squared_map_2d == pytest.approx(chi_squared_map_2d,
                                                                     1.0e-4)

        chi_squared_normal = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_normal.chi_squared_map_2d, mask=ld_blur.mask_2d)
        chi_squared_sensitive = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=fit.fit_sensitive.chi_squared_map_2d, mask=ld_blur.mask_2d)
        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=ld_blur.mask_2d, noise_map=ld_blur.noise_map_2d)

        assert fit.fit_normal.likelihood == -0.5 * (
                chi_squared_normal + noise_normalization)
        assert fit.fit_sensitive.likelihood == -0.5 * (
                chi_squared_sensitive + noise_normalization)

        assert fit.figure_of_merit == fit.fit_sensitive.likelihood - fit.fit_normal.likelihood

        fit_from_factory = sensitivity_fit.fit_lens_data_with_sensitivity_tracers(
            lens_data=ld_blur,
            tracer_normal=tracer_normal,
            tracer_sensitive=tracer_sensitive)

        assert fit.figure_of_merit == fit_from_factory.figure_of_merit
