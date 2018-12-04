import numpy as np
import pytest

from autolens.data.fitting.util import fitting_util
from autolens.data.array import mask as msk, scaled_array as sca
from autolens.model.galaxy import galaxy_data
from autolens.model.galaxy import galaxy as g, galaxy_fitting
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.mock.mock_galaxy import MockGalaxy


class TestGalaxyFit:

    class TestLikelihood:

        def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise_term(self):

            array = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            noise_map = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyDataIntensities(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data_y = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            g_data_x = galaxy_data.GalaxyDataDeflectionsX(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFitDeflections(galaxy_datas=[g_data_y, g_data_x], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.likelihoods[0] == -0.5 * np.log(2 * np.pi * 1.0)
            assert fit.likelihoods[1] == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__noise_not_1__all_terms_correct(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0*np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyDataIntensities(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data_y = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            g_data_x = galaxy_data.GalaxyDataDeflectionsX(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data_y, g_data_x], model_galaxy=g0)
            assert fit.model_galaxy == g0
            assert fit.chi_squared_terms[0] == (25.0 / 4.0)
            assert fit.reduced_chi_squareds[0] == (25.0 / 4.0) / 2.0
            assert fit.likelihoods[0] == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))
            assert fit.chi_squared_terms[1] == (25.0 / 4.0)
            assert fit.reduced_chi_squareds[1] == (25.0 / 4.0) / 2.0
            assert fit.likelihoods[1] == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

        def test__same_as_above_but_fast_likelihood(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0*np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyDataIntensities(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data], model_galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_datas=[g_data], galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data], model_galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_datas=[g_data], galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data], model_galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_datas=[g_data], galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data_y = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            g_data_x = galaxy_data.GalaxyDataDeflectionsX(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data_y, g_data_x], model_galaxy=g0) == \
                   2.0 * -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

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
            g_data = galaxy_data.GalaxyDataIntensities(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data = galaxy.intensities_from_grid(grid=g_data.grids.sub)
            model_datas = [g_data.grids.sub.sub_data_to_regular_data(sub_array=model_data)]
            residuals = fitting_util.residuals_from_datas_and_model_datas([g_data[:]], model_datas)
            chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_maps(residuals, [g_data.noise_map_])

            assert g_data.grids.regular.scaled_array_from_array_1d(g_data.noise_map_) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(model_datas[0]) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
                   pytest.approx(fit.residual, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
                   pytest.approx(fit.chi_squared, 1e-4)

            chi_squared_terms = fitting_util.chi_squared_terms_from_chi_squareds(chi_squareds)
            noise_terms = fitting_util.noise_terms_from_noise_maps([g_data.noise_map_])
            likelihoods = fitting_util.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms)

            assert likelihoods[0] == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data],
                                                                                         model_galaxy=galaxy)
            assert fast_likelihood == pytest.approx(fit.likelihood)

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
            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data = galaxy.surface_density_from_grid(grid=g_data.grids.sub)
            model_datas = [g_data.grids.sub.sub_data_to_regular_data(sub_array=model_data)]
            residuals = fitting_util.residuals_from_datas_and_model_datas([g_data[:]], model_datas)
            chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_maps(residuals, [g_data.noise_map_])

            assert g_data.grids.regular.scaled_array_from_array_1d(g_data.noise_map_) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(model_datas[0]) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
                   pytest.approx(fit.residual, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
                   pytest.approx(fit.chi_squared, 1e-4)

            chi_squared_terms = fitting_util.chi_squared_terms_from_chi_squareds(chi_squareds)
            noise_terms = fitting_util.noise_terms_from_noise_maps([g_data.noise_map_])
            likelihoods = fitting_util.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms)

            assert likelihoods[0] == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data],
                                                                                         model_galaxy=galaxy)
            assert fast_likelihood == pytest.approx(fit.likelihood)
            
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
            g_data = galaxy_data.GalaxyDataPotential(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data], model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data = galaxy.potential_from_grid(grid=g_data.grids.sub)
            model_datas = [g_data.grids.sub.sub_data_to_regular_data(sub_array=model_data)]
            residuals = fitting_util.residuals_from_datas_and_model_datas([g_data[:]], model_datas)
            chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_maps(residuals, [g_data.noise_map_])

            assert g_data.grids.regular.scaled_array_from_array_1d(g_data.noise_map_) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(model_datas[0]) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
                   pytest.approx(fit.residual, 1e-4)
            assert g_data.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
                   pytest.approx(fit.chi_squared, 1e-4)

            chi_squared_terms = fitting_util.chi_squared_terms_from_chi_squareds(chi_squareds)
            noise_terms = fitting_util.noise_terms_from_noise_maps([g_data.noise_map_])
            likelihoods = fitting_util.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms)

            assert likelihoods[0] == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[g_data],
                                                                                         model_galaxy=galaxy)
            assert fast_likelihood == pytest.approx(fit.likelihood)
            
        
        def test__deflections(self):

            im_y = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 5.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)

            im_x = sca.ScaledSquarePixelArray(array=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 2.0, 3.0, 0.0],
                                                            [0.0, 4.0, 0.0, 6.0, 0.0],
                                                            [0.0, 7.0, 8.0, 9.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, False, False, False, True],
                                            [True, True, True, True, True]]), pixel_scale=1.0)
            noise_map = 2.0 * np.ones((5, 5))

            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            g_data_y = galaxy_data.GalaxyDataDeflectionsY(array=im_y, mask=mask, noise_map=noise_map, sub_grid_size=2)
            g_data_x = galaxy_data.GalaxyDataDeflectionsX(array=im_x, mask=mask, noise_map=noise_map, sub_grid_size=2)

            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[g_data_y, g_data_x], model_galaxy=galaxy)

            assert fit.model_galaxy == galaxy

            model_data_y = galaxy.deflections_from_grid(grid=g_data_y.grids.sub)
            model_data_y = g_data_y.grids.sub.sub_data_to_regular_data(sub_array=model_data_y[:, 0])
            model_data_x = galaxy.deflections_from_grid(grid=g_data_y.grids.sub)
            model_data_x = g_data_y.grids.sub.sub_data_to_regular_data(sub_array=model_data_x[:, 1])
            model_datas = [model_data_y, model_data_x]
            residuals = fitting_util.residuals_from_datas_and_model_datas([g_data_y[:], g_data_x[:]], model_datas)
            chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_maps(residuals, [g_data_y.noise_map_,
                                                                                          g_data_x.noise_map_])

            assert g_data_y.grids.regular.scaled_array_from_array_1d(g_data_y.noise_map_) == \
                   pytest.approx(fit.noise_maps[0], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(model_data_y) == \
                   pytest.approx(fit.model_datas[0], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
                   pytest.approx(fit.residuals[0], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
                   pytest.approx(fit.chi_squareds[0], 1e-4)

            assert g_data_y.grids.regular.scaled_array_from_array_1d(g_data_x.noise_map_) == \
                   pytest.approx(fit.noise_maps[1], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(model_data_x) == \
                   pytest.approx(fit.model_datas[1], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(residuals[1]) == \
                   pytest.approx(fit.residuals[1], 1e-4)
            assert g_data_y.grids.regular.scaled_array_from_array_1d(chi_squareds[1]) == \
                   pytest.approx(fit.chi_squareds[1], 1e-4)

            chi_squared_terms = fitting_util.chi_squared_terms_from_chi_squareds(chi_squareds)
            noise_terms = fitting_util.noise_terms_from_noise_maps([g_data_y.noise_map_, g_data_x.noise_map_])
            likelihoods = fitting_util.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms)

            assert likelihoods[0] == pytest.approx(fit.likelihoods[0], 1e-4)
            assert likelihoods[1] == pytest.approx(fit.likelihoods[1], 1e-4)
            assert likelihoods[0] + likelihoods[1] == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(
                galaxy_datas=[g_data_y, g_data_x], model_galaxy=galaxy)
            assert fast_likelihood == pytest.approx(fit.likelihood)