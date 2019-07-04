import numpy as np
import pytest

import autofit as af
from autolens.data.array import mask as msk, scaled_array as sca
from autolens.model.galaxy import galaxy as g, galaxy_fit
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.unit.mock.model.mock_galaxy import MockGalaxy


class TestGalaxyFit:

    class TestLikelihood:

        def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise(self):
            image = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            noise_map = sca.ScaledSquarePixelArray(array=np.ones((3, 3)),
                                                   pixel_scale=1.0)

            galaxy_data = gd.GalaxyData(image=image, noise_map=noise_map,
                                        pixel_scale=3.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                            [True, False, True],
                                            [True, True, True]]), pixel_scale=1.0)
            g0 = MockGalaxy(value=1.0)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_intensities=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_convergence=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_potential=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_deflections_y=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_deflections_x=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__noise_not_1__alls_correct(self):
            image = sca.ScaledSquarePixelArray(array=5.0 * np.ones((3, 4)),
                                               pixel_scale=1.0)
            image[1, 2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0 * np.ones((3, 4)),
                                                   pixel_scale=1.0)

            galaxy_data = gd.GalaxyData(image=image, noise_map=noise_map,
                                        pixel_scale=3.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                            [True, False, False, True],
                                            [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0, shape=2)

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_intensities=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])

            assert fit.model_galaxies == [g0]
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * (
                        (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2))

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_convergence=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * (
                        (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2))

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_potential=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.model_galaxies == [g0]
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * (
                        (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2))

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_deflections_y=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * (
                        (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2))

            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask,
                                               sub_grid_size=1,
                                               use_deflections_x=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * (
                        (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2))

    class TestCompareToManual:

        def test__intensities(self, gal_data_5x5, mask_5x5):
            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5,
                                               sub_grid_size=2,
                                               use_intensities=True)

            galaxy = g.Galaxy(redshift=0.5, light=lp.SphericalSersic(centre=(1.0, 2.0),
                                                                     intensity=1.0))
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data,
                                       model_galaxies=[galaxy])

            assert fit.model_galaxies == [galaxy]

            model_data_1d = galaxy.intensities_from_grid(
                grid=galaxy_fit_data.grid_stack.sub)
            model_data_1d = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
                sub_array_1d=model_data_1d)
            model_data_2d = galaxy_fit_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
                data=galaxy_fit_data.image_2d, mask=galaxy_fit_data.mask_2d,
                model_data=model_data_2d)

            assert residual_map_2d == pytest.approx(fit.residual_map_2d, 1e-4)

            chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_2d, mask=galaxy_fit_data.mask_2d,
                noise_map=galaxy_fit_data.noise_map_2d)

            assert chi_squared_map_2d == pytest.approx(fit.chi_squared_map_2d, 1e-4)

            chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
                chi_squared_map=chi_squared_map_2d, mask=mask_5x5)

            noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
                mask=galaxy_fit_data.mask_2d, noise_map=galaxy_fit_data.noise_map_2d)

            likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
                chi_squared=chi_squared, noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__convergence(self, gal_data_5x5, mask_5x5):
            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5,
                                               sub_grid_size=2,
                                               use_convergence=True)

            galaxy = g.Galaxy(redshift=0.5,
                              mass=mp.SphericalIsothermal(centre=(1.0, 2.0),
                                                          einstein_radius=1.0))
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data,
                                       model_galaxies=[galaxy])

            assert fit.model_galaxies == [galaxy]

            model_data_1d = galaxy.convergence_from_grid(
                grid=galaxy_fit_data.grid_stack.sub)
            model_data_1d = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
                sub_array_1d=model_data_1d)
            model_data_2d = galaxy_fit_data.map_to_scaled_array(array_1d=model_data_1d)

            residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
                data=galaxy_fit_data.image_2d, mask=galaxy_fit_data.mask_2d,
                model_data=model_data_2d)
            assert residual_map_2d == pytest.approx(fit.residual_map_2d, 1e-4)

            chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_2d, mask=galaxy_fit_data.mask_2d,
                noise_map=galaxy_fit_data.noise_map_2d)
            assert chi_squared_map_2d == pytest.approx(fit.chi_squared_map_2d, 1e-4)

            chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
                chi_squared_map=chi_squared_map_2d, mask=mask_5x5)

            noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
                mask=galaxy_fit_data.mask_2d, noise_map=galaxy_fit_data.noise_map_2d)

            likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
                chi_squared=chi_squared, noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__potential(self, gal_data_5x5, mask_5x5):
            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5,
                                               sub_grid_size=2,
                                               use_potential=True)

            galaxy = g.Galaxy(redshift=0.5,
                              mass=mp.SphericalIsothermal(centre=(1.0, 2.0),
                                                          einstein_radius=1.0))

            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data,
                                       model_galaxies=[galaxy])

            assert fit.model_galaxies == [galaxy]

            model_data_1d = galaxy.potential_from_grid(
                grid=galaxy_fit_data.grid_stack.sub)
            model_data_1d = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
                sub_array_1d=model_data_1d)
            model_data_2d = galaxy_fit_data.map_to_scaled_array(array_1d=model_data_1d)

            residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
                data=galaxy_fit_data.image_2d, mask=galaxy_fit_data.mask_2d,
                model_data=model_data_2d)

            assert residual_map_2d == pytest.approx(fit.residual_map_2d, 1e-4)

            chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_2d, mask=galaxy_fit_data.mask_2d,
                noise_map=galaxy_fit_data.noise_map_2d)

            assert chi_squared_map_2d == pytest.approx(fit.chi_squared_map_2d, 1e-4)

            chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
                chi_squared_map=chi_squared_map_2d, mask=mask_5x5)

            noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
                mask=galaxy_fit_data.mask_2d, noise_map=galaxy_fit_data.noise_map_2d)

            likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
                chi_squared=chi_squared, noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__deflections_y(self, gal_data_5x5, mask_5x5):
            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5,
                                               sub_grid_size=2,
                                               use_deflections_y=True)

            galaxy = g.Galaxy(redshift=0.5,
                              mass=mp.SphericalIsothermal(centre=(1.0, 2.0),
                                                          einstein_radius=1.0))

            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data,
                                       model_galaxies=[galaxy])

            assert fit.model_galaxies == [galaxy]

            model_data_1d = galaxy.deflections_from_grid(
                grid=galaxy_fit_data.grid_stack.sub)
            model_data_1d = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
                sub_array_1d=model_data_1d[:, 0])
            model_data_2d = galaxy_fit_data.map_to_scaled_array(array_1d=model_data_1d)

            residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
                data=galaxy_fit_data.image_2d, mask=galaxy_fit_data.mask_2d,
                model_data=model_data_2d)

            assert residual_map_2d == pytest.approx(fit.residual_map_2d, 1e-4)

            chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_2d, mask=galaxy_fit_data.mask_2d,
                noise_map=galaxy_fit_data.noise_map_2d)

            assert chi_squared_map_2d == pytest.approx(fit.chi_squared_map_2d, 1e-4)

            chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
                chi_squared_map=chi_squared_map_2d, mask=mask_5x5)

            noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
                mask=galaxy_fit_data.mask_2d, noise_map=galaxy_fit_data.noise_map_2d)

            likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
                chi_squared=chi_squared, noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__deflections_x(self, gal_data_5x5, mask_5x5):
            galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5,
                                               sub_grid_size=2,
                                               use_deflections_x=True)

            galaxy = g.Galaxy(redshift=0.5,
                              mass=mp.SphericalIsothermal(centre=(1.0, 2.0),
                                                          einstein_radius=1.0))
            fit = galaxy_fit.GalaxyFit(galaxy_data=galaxy_fit_data,
                                       model_galaxies=[galaxy])

            assert fit.model_galaxies == [galaxy]

            model_data_1d = galaxy.deflections_from_grid(
                grid=galaxy_fit_data.grid_stack.sub)
            model_data_1d = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
                sub_array_1d=model_data_1d[:, 1])
            model_data_2d = galaxy_fit_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
                data=galaxy_fit_data.image_2d, mask=galaxy_fit_data.mask_2d,
                model_data=model_data_2d)

            assert residual_map_2d == pytest.approx(fit.residual_map_2d, 1e-4)

            chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_2d, mask=galaxy_fit_data.mask_2d,
                noise_map=galaxy_fit_data.noise_map_2d)

            assert chi_squared_map_2d == pytest.approx(fit.chi_squared_map_2d, 1e-4)

            chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
                chi_squared_map=chi_squared_map_2d, mask=mask_5x5)

            noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
                mask=galaxy_fit_data.mask_2d, noise_map=galaxy_fit_data.noise_map_2d)

            likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
                chi_squared=chi_squared, noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)
