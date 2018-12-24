import numpy as np
import pytest

from autofit.core import fitting_util
from autolens.data.array import mask as msk, scaled_array as sca
from autolens.model.galaxy import galaxy_data
from autolens.model.galaxy import galaxy as g, galaxy_fit
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.mock.mock_galaxy import MockGalaxy


class TestGalaxyFit:

    class TestLikelihood:

        def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise(self):

            array = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            noise_map = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)
            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_intensities=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_surface_density=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_potential=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                              use_deflections_y=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                              use_deflections_x=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__noise_not_1__alls_correct(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0*np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0, shape=2)

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_intensities=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)

            assert fit.model_galaxy == g0
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_surface_density=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                            use_potential=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                              use_deflections_y=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1,
                                              use_deflections_x=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=g0)
            assert fit.chi_squared == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

    class TestCompareToManual:

        def test__intensities(self):

            im = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)
            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0))
            g_data = galaxy_data.GalaxyData(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2,
                                            use_intensities=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_1d = galaxy.intensities_from_grid(grid=g_data.grid_stack.sub)
            model_data_1d = g_data.grid_stack.sub.sub_data_to_regular_data(sub_array=model_data_1d)
            model_data = g_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=g_data.array, mask=g_data.mask,
                                                                                   model_data=model_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map, 
                                                                                 mask=mask)
            noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=g_data.mask,
                                                                                  noise_map=g_data.noise_map)
            likelihood = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__surface_density(self):

            im = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)
            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            g_data = galaxy_data.GalaxyData(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2,
                                            use_surface_density=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_1d = galaxy.surface_density_from_grid(grid=g_data.grid_stack.sub)
            model_data_1d = g_data.grid_stack.sub.sub_data_to_regular_data(sub_array=model_data_1d)
            model_data = g_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=g_data.array, mask=g_data.mask,
                                                                                   model_data=model_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=mask)
            noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=g_data.mask,
                                                                                  noise_map=g_data.noise_map)
            likelihood = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)
            
        def test__potential(self):

            im = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)
            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            g_data = galaxy_data.GalaxyData(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2,
                                            use_potential=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_1d = galaxy.potential_from_grid(grid=g_data.grid_stack.sub)
            model_data_1d = g_data.grid_stack.sub.sub_data_to_regular_data(sub_array=model_data_1d)
            model_data = g_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=g_data.array, mask=g_data.mask,
                                                                                   model_data=model_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=mask)
            noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=g_data.mask,
                                                                                  noise_map=g_data.noise_map)
            likelihood = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__deflections_y(self):

            im = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)
            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            g_data = galaxy_data.GalaxyData(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2,
                                            use_deflections_y=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_1d = galaxy.deflections_from_grid(grid=g_data.grid_stack.sub)
            model_data_1d = g_data.grid_stack.sub.sub_data_to_regular_data(sub_array=model_data_1d[:,0])
            model_data = g_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=g_data.array, mask=g_data.mask,
                                                                                   model_data=model_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=g_data.mask,
                                                                                                noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=g_data.mask,
                                                                                                noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map, mask=mask)
            noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=g_data.mask,
                                                                                           noise_map=g_data.noise_map)
            likelihood = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

        def test__deflections_x(self):

            im = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)
            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            g_data = galaxy_data.GalaxyData(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2,
                                            use_deflections_x=True)
            fit = galaxy_fit.GalaxyFit(galaxy_data=g_data, model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_1d = galaxy.deflections_from_grid(grid=g_data.grid_stack.sub)
            model_data_1d = g_data.grid_stack.sub.sub_data_to_regular_data(sub_array=model_data_1d[:,1])
            model_data = g_data.map_to_scaled_array(array_1d=model_data_1d)
            residual_map = fitting_util.residual_map_from_data_mask_and_model_data(data=g_data.array, mask=g_data.mask,
                                                                                   model_data=model_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared_map = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=g_data.mask, noise_map=g_data.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=mask)
            noise_normalization = fitting_util.noise_normalization_from_noise_map_and_mask(mask=g_data.mask,
                                                                                  noise_map=g_data.noise_map)
            likelihood = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)
