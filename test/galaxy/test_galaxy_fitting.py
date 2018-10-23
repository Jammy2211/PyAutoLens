import numpy as np
import pytest

from autolens.fitting import fitting
from autolens.imaging import scaled_array as sca
from autolens.imaging import mask as msk
from autolens.galaxy import galaxy_data
from autolens.galaxy import galaxy as g
from autolens.galaxy import galaxy_fitting
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
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
            fit = galaxy_fitting.GalaxyFit(galaxy_data=g_data, galaxy=g0)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_data=g_data, galaxy=g0)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_data=g_data, galaxy=g0)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_data=g_data, galaxy=g0)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

            g_data = galaxy_data.GalaxyDataDeflectionsX(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.GalaxyFit(galaxy_data=g_data, galaxy=g0)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__noise_not_1__all_terms_correct(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0*np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyDataIntensities(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=g0)
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=g0)
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=g0)
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=g0)
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataDeflectionsX(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=g0)
            assert fit.chi_squared_term == (25.0 / 4.0)
            assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 / 4.0) + 2.0*np.log(2 * np.pi * 2.0**2))

        def test__same_as_above_but_fast_likelihood(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=2.0*np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            g0 = MockGalaxy(value=1.0)

            g_data = galaxy_data.GalaxyDataIntensities(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataPotential(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFit.fast_likelihood(galaxy_data=g_data, galaxy=g0) == \
                   -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))

            g_data = galaxy_data.GalaxyDataDeflectionsY(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)
            assert galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data, galaxy=g0) == \
                   2.0 * -0.5 * (25.0 / 4.0 + 2.0*np.log(2 * np.pi * 2.0**2))
            assert galaxy_fitting.GalaxyFitDeflectionsY.fast_likelihood(galaxy_data_y=g_data, galaxy_data_x=g_data,
                                                                        galaxy=g0) == \
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

            g_data = galaxy_data.GalaxyDataIntensities(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0))
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=galaxy)

            model_data = galaxy.intensities_from_grid(grid=g_data.grids.sub)
            model_data = g_data.grids.sub.sub_data_to_image(sub_array=model_data)
            residuals = fitting.residuals_from_data_and_model(g_data[:], model_data)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, g_data.noise_map)

            assert g_data.grids.image.scaled_array_from_array_1d(g_data.noise_map) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(model_data) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(g_data.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data,
                                                                                         galaxy=galaxy)
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
            noise_map = 2.0*np.ones((5,5))

            g_data = galaxy_data.GalaxyDataSurfaceDensity(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=galaxy)

            model_data = galaxy.surface_density_from_grid(grid=g_data.grids.sub)
            model_data = g_data.grids.sub.sub_data_to_image(sub_array=model_data)
            residuals = fitting.residuals_from_data_and_model(g_data[:], model_data)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, g_data.noise_map)

            assert g_data.grids.image.scaled_array_from_array_1d(g_data.noise_map) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(model_data) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit.chi_squareds, 1e-4)
            
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(g_data.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data,
                                                                                         galaxy=galaxy)
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

            g_data = galaxy_data.GalaxyDataPotential(array=im, mask=mask, noise_map=noise_map, sub_grid_size=2)
            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            fit = galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_data=g_data, galaxy=galaxy)

            model_data = galaxy.potential_from_grid(grid=g_data.grids.sub)
            model_data = g_data.grids.sub.sub_data_to_image(sub_array=model_data)
            residuals = fitting.residuals_from_data_and_model(g_data[:], model_data)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, g_data.noise_map)

            assert g_data.grids.image.scaled_array_from_array_1d(g_data.noise_map) == \
                   pytest.approx(fit.noise_map, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(model_data) == \
                   pytest.approx(fit.model_data, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit.residuals, 1e-4)
            assert g_data.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(g_data.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data=g_data,
                                                                                         galaxy=galaxy)
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

            g_data_y = galaxy_data.GalaxyDataDeflectionsY(array=im_y, mask=mask, noise_map=noise_map, sub_grid_size=2)
            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            fit_y = galaxy_fitting.GalaxyFitDeflectionsY(galaxy_data=g_data_y, galaxy=galaxy)

            model_data = galaxy.deflections_from_grid(grid=g_data_y.grids.sub)
            model_data_y = g_data_y.grids.sub.sub_data_to_image(sub_array=model_data[:,0])
            residuals = fitting.residuals_from_data_and_model(g_data_y[:], model_data_y)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, g_data_y.noise_map)

            assert g_data_y.grids.image.scaled_array_from_array_1d(g_data_y.noise_map) == \
                   pytest.approx(fit_y.noise_map, 1e-4)
            assert g_data_y.grids.image.scaled_array_from_array_1d(model_data_y) == \
                   pytest.approx(fit_y.model_data, 1e-4)
            assert g_data_y.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit_y.residuals, 1e-4)
            assert g_data_y.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit_y.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(g_data_y.noise_map)
            likelihood_y = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood_y == pytest.approx(fit_y.likelihood, 1e-4)

            g_data_x = galaxy_data.GalaxyDataDeflectionsX(array=im_x, mask=mask, noise_map=noise_map, sub_grid_size=2)
            galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0))
            fit_x = galaxy_fitting.GalaxyFitDeflectionsX(galaxy_data=g_data_x, galaxy=galaxy)

            model_data = galaxy.deflections_from_grid(grid=g_data_x.grids.sub)
            model_data_x = g_data_x.grids.sub.sub_data_to_image(sub_array=model_data[:,1])
            residuals = fitting.residuals_from_data_and_model(g_data_x[:], model_data_x)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, g_data_x.noise_map)

            assert g_data_x.grids.image.scaled_array_from_array_1d(g_data_x.noise_map) == \
                   pytest.approx(fit_x.noise_map, 1e-4)
            assert g_data_x.grids.image.scaled_array_from_array_1d(model_data_x) == \
                   pytest.approx(fit_x.model_data, 1e-4)
            assert g_data_x.grids.image.scaled_array_from_array_1d(residuals) == \
                   pytest.approx(fit_x.residuals, 1e-4)
            assert g_data_x.grids.image.scaled_array_from_array_1d(chi_squareds) == \
                   pytest.approx(fit_x.chi_squareds, 1e-4)

            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise_map(g_data_x.noise_map)
            likelihood_x = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert likelihood_x == pytest.approx(fit_x.likelihood, 1e-4)

            fast_likelihood = galaxy_fitting.GalaxyFitDeflectionsY.fast_likelihood(galaxy_data_y=g_data_y,
                                                                                   galaxy_data_x=g_data_x, galaxy=galaxy)

            assert fast_likelihood == pytest.approx(likelihood_y + likelihood_x)