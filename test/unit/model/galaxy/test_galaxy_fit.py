import autolens as al
import numpy as np
import pytest

import autofit as af
from test.unit.mock.model.mock_galaxy import MockGalaxy


class TestLikelihood:
    def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise(self):
        image = al.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

        noise_map = al.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

        galaxy_data = al.GalaxyData(image=image, noise_map=noise_map, pixel_scale=3.0)

        mask = al.Mask(
            array=np.array(
                [[True, True, True], [True, False, True], [True, True, True]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )
        g0 = MockGalaxy(value=1.0)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

    def test__1x2_image__noise_not_1__alls_correct(self):
        image = al.ScaledSquarePixelArray(array=5.0 * np.ones((3, 4)), pixel_scale=1.0)
        image[1, 2] = 4.0

        noise_map = al.ScaledSquarePixelArray(
            array=2.0 * np.ones((3, 4)), pixel_scale=1.0
        )

        galaxy_data = al.GalaxyData(image=image, noise_map=noise_map, pixel_scale=3.0)

        mask = al.Mask(
            array=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        g0 = MockGalaxy(value=1.0, shape=2)

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])

        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[g0])
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )


class TestCompareToManual:
    def test__image(self, gal_data_7x7, sub_mask_7x7):
        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            light=al.light_profiles.SphericalSersic(centre=(1.0, 2.0), intensity=1.0),
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[galaxy])

        assert fit.model_galaxies == [galaxy]

        model_data_2d = galaxy.profile_image_from_grid(
            grid=galaxy_fit_data.grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=galaxy_fit_data.image(return_in_2d=True),
            mask=galaxy_fit_data.mask,
            model_data=model_data_2d,
        )

        assert residual_map_2d == pytest.approx(
            fit.residual_map(return_in_2d=True), 1e-4
        )

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        assert chi_squared_map_2d == pytest.approx(
            fit.chi_squared_map(return_in_2d=True), 1e-4
        )

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=sub_mask_7x7
        )

        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__convergence(self, gal_data_7x7, sub_mask_7x7):
        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.SphericalIsothermal(
                centre=(1.0, 2.0), einstein_radius=1.0
            ),
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[galaxy])

        assert fit.model_galaxies == [galaxy]

        model_data_2d = galaxy.convergence_from_grid(
            grid=galaxy_fit_data.grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=galaxy_fit_data.image(return_in_2d=True),
            mask=galaxy_fit_data.mask,
            model_data=model_data_2d,
        )
        assert residual_map_2d == pytest.approx(
            fit.residual_map(return_in_2d=True), 1e-4
        )

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )
        assert chi_squared_map_2d == pytest.approx(
            fit.chi_squared_map(return_in_2d=True), 1e-4
        )

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=sub_mask_7x7
        )

        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__potential(self, gal_data_7x7, sub_mask_7x7):
        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.SphericalIsothermal(
                centre=(1.0, 2.0), einstein_radius=1.0
            ),
        )

        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[galaxy])

        assert fit.model_galaxies == [galaxy]

        model_data_2d = galaxy.potential_from_grid(
            grid=galaxy_fit_data.grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=galaxy_fit_data.image(return_in_2d=True),
            mask=galaxy_fit_data.mask,
            model_data=model_data_2d,
        )

        assert residual_map_2d == pytest.approx(
            fit.residual_map(return_in_2d=True), 1e-4
        )

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        assert chi_squared_map_2d == pytest.approx(
            fit.chi_squared_map(return_in_2d=True), 1e-4
        )

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=sub_mask_7x7
        )

        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__deflections_y(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.SphericalIsothermal(
                centre=(1.0, 2.0), einstein_radius=1.0
            ),
        )

        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[galaxy])

        assert fit.model_galaxies == [galaxy]

        model_data_2d = galaxy.deflections_from_grid(
            grid=galaxy_fit_data.grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )[:, :, 0]

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=galaxy_fit_data.image(return_in_2d=True),
            mask=galaxy_fit_data.mask,
            model_data=model_data_2d,
        )

        assert residual_map_2d == pytest.approx(
            fit.residual_map(return_in_2d=True), 1e-4
        )

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        assert chi_squared_map_2d == pytest.approx(
            fit.chi_squared_map(return_in_2d=True), 1e-4
        )

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=sub_mask_7x7
        )

        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__deflections_x(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.SphericalIsothermal(
                centre=(1.0, 2.0), einstein_radius=1.0
            ),
        )
        fit = al.GalaxyFit(galaxy_data=galaxy_fit_data, model_galaxies=[galaxy])

        assert fit.model_galaxies == [galaxy]

        model_data_2d = galaxy.deflections_from_grid(
            grid=galaxy_fit_data.grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )[:, :, 1]

        residual_map_2d = af.fit_util.residual_map_from_data_mask_and_model_data(
            data=galaxy_fit_data.image(return_in_2d=True),
            mask=galaxy_fit_data.mask,
            model_data=model_data_2d,
        )

        assert residual_map_2d == pytest.approx(
            fit.residual_map(return_in_2d=True), 1e-4
        )

        chi_squared_map_2d = af.fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_2d,
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        assert chi_squared_map_2d == pytest.approx(
            fit.chi_squared_map(return_in_2d=True), 1e-4
        )

        chi_squared = af.fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map_2d, mask=sub_mask_7x7
        )

        noise_normalization = af.fit_util.noise_normalization_from_noise_map_and_mask(
            mask=galaxy_fit_data.mask,
            noise_map=galaxy_fit_data.noise_map(return_in_2d=True),
        )

        likelihood = af.fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)
