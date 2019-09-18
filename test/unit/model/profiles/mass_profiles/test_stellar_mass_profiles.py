import numpy as np
import pytest

import autolens as al

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestSersic(object):
    def test__constructor_and_units(self):
        sersic = al.mass_profiles.EllipticalSersic(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, al.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = al.mass_profiles.SphericalSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, al.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        sersic = al.mass_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]]), bypass_decorator=True
        ) == pytest.approx(4.90657319276, 1e-3)

        sersic = al.mass_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = al.mass_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(5.38066670129, 1e-3)

    def test__deflections_correct_values(self):
        sersic = al.mass_profiles.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)

        sersic = al.mass_profiles.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)
        assert deflections[1, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[1, 1] == pytest.approx(0.79374, 1e-3)

    def test__surfce_density__change_geometry(self):
        sersic_0 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = al.mass_profiles.SphericalSersic(centre=(1.0, 1.0))
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 1.0]]), bypass_decorator=True
        ) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 0.0]]), bypass_decorator=True
        )

        sersic_0 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        )

        sersic_0 = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0
        )
        sersic_1 = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0
        )
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        )

    def test__deflections__change_geometry(self):
        sersic_0 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = al.mass_profiles.SphericalSersic(centre=(1.0, 1.0))
        deflections_0 = sersic_0.deflections_from_grid(
            grid=np.array([[1.0, 1.0]]), bypass_decorator=True
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=np.array([[0.0, 0.0]]), bypass_decorator=True
        )
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        sersic_0 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = al.mass_profiles.SphericalSersic(centre=(0.0, 0.0))
        deflections_0 = sersic_0.deflections_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        )
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

        sersic_0 = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0
        )
        sersic_1 = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0
        )
        deflections_0 = sersic_0.deflections_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        )
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__spherical_and_elliptical_identical(self):
        elliptical = al.mass_profiles.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        spherical = al.mass_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid, bypass_decorator=True)
            == spherical.convergence_from_grid(grid=grid, bypass_decorator=True)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid, bypass_decorator=True) == spherical.potential_from_grid(grid=grid, bypass_decorator=True)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid, bypass_decorator=True),
            spherical.deflections_from_grid(grid=grid, bypass_decorator=True),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = al.mass_profiles.EllipticalSersic(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = al.mass_profiles.SphericalSersic(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        sersic = al.mass_profiles.EllipticalSersic()

        convergence = sersic.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        sersic = al.mass_profiles.SphericalSersic()

        convergence = sersic.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestExponential(object):
    def test__constructor_and_units(self):
        exponential = al.mass_profiles.EllipticalExponential(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], al.Length)
        assert isinstance(exponential.centre[1], al.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, al.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, al.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, al.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = al.mass_profiles.SphericalExponential(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], al.Length)
        assert isinstance(exponential.centre[1], al.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, al.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, al.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, al.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        exponential = al.mass_profiles.EllipticalExponential(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(4.9047, 1e-3)

        exponential = al.mass_profiles.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(4.8566, 1e-3)

        exponential = al.mass_profiles.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = al.mass_profiles.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = al.mass_profiles.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(4.8566, 1e-3)

    def test__deflections_correct_values(self):
        exponential = al.mass_profiles.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

        exponential = al.mass_profiles.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = al.mass_profiles.EllipticalExponential(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = al.mass_profiles.SphericalExponential(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid, bypass_decorator=True)
            == spherical.convergence_from_grid(grid=grid, bypass_decorator=True)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid, bypass_decorator=True) == spherical.potential_from_grid(grid=grid, bypass_decorator=True)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid, bypass_decorator=True),
            spherical.deflections_from_grid(grid=grid, bypass_decorator=True),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = al.mass_profiles.EllipticalExponential(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = al.mass_profiles.SphericalExponential(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        exponential = al.mass_profiles.EllipticalExponential()

        convergence = exponential.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        exponential = al.mass_profiles.SphericalExponential()

        convergence = exponential.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestDevVaucouleurs(object):
    def test__constructor_and_units(self):
        dev_vaucouleurs = al.mass_profiles.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], al.Length)
        assert isinstance(dev_vaucouleurs.centre[1], al.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, al.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, al.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, al.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = al.mass_profiles.SphericalDevVaucouleurs(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], al.Length)
        assert isinstance(dev_vaucouleurs.centre[1], al.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, al.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, al.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, al.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(5.6697, 1e-3)

        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(7.4455, 1e-3)

        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 7.4455, 1e-3)

        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert dev.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 7.4455, 1e-3)

        sersic = al.mass_profiles.SphericalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(0.351797, 1e-3)

    def test__deflections_correct_values(self):
        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )
        deflections = dev.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(-24.528, 1e-3)
        assert deflections[0, 1] == pytest.approx(-3.37605, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = al.mass_profiles.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = al.mass_profiles.SphericalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid, bypass_decorator=True)
            == spherical.convergence_from_grid(grid=grid, bypass_decorator=True)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid, bypass_decorator=True) == spherical.potential_from_grid(grid=grid, bypass_decorator=True)

        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid, bypass_decorator=True),
            spherical.deflections_from_grid(grid=grid, bypass_decorator=True),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = al.mass_profiles.SphericalDevVaucouleurs(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        dev_vaucouleurs = al.mass_profiles.EllipticalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        dev_vaucouleurs = al.mass_profiles.SphericalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestSersicMassRadialGradient(object):
    def test__constructor_and_units(self):
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, al.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = al.mass_profiles.SphericalSersicRadialGradient(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, al.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(0.6 * 0.351797, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]]), bypass_decorator=True
        ) == pytest.approx(0.75 * 4.90657319276, 1e-3)

        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(2.836879 * 5.38066670129, abs=2e-01)

    def test__deflections_correct_values(self):
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(3.60324873535244, 1e-3)
        assert deflections[0, 1] == pytest.approx(2.3638898009652, 1e-3)

        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )
        assert deflections[0, 0] == pytest.approx(0.97806399756448, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.725459334118341, 1e-3)

    def test__compare_to_sersic(self):
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=1.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        exponential = al.mass_profiles.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        exponential_deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        assert (
            sersic_deflections[0, 0]
            == exponential_deflections[0, 0]
            == pytest.approx(0.90493, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == exponential_deflections[0, 1]
            == pytest.approx(0.62569, 1e-3)
        )

        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            sersic_index=4.0,
            mass_to_light_ratio=3.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        dev = al.mass_profiles.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )

        dev_deflections = dev.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        assert (
            sersic_deflections[0, 0]
            == dev_deflections[0, 0]
            == pytest.approx(-24.528, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == dev_deflections[0, 1]
            == pytest.approx(-3.37605, 1e-3)
        )

        sersic_grad = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_grad_deflections = sersic_grad.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        sersic = al.mass_profiles.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]]), bypass_decorator=True
        )

        assert (
            sersic_grad_deflections[0, 0]
            == sersic_deflections[0, 0]
            == pytest.approx(1.1446, 1e-3)
        )
        assert (
            sersic_grad_deflections[0, 1]
            == sersic_deflections[0, 1]
            == pytest.approx(0.79374, 1e-3)
        )

    def test__spherical_and_elliptical_identical(self):
        elliptical = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        spherical = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert (
            elliptical.convergence_from_grid(grid=grid, bypass_decorator=True)
            == spherical.convergence_from_grid(grid=grid, bypass_decorator=True)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid, bypass_decorator=True) == spherical.potential_from_grid(grid=grid, bypass_decorator=True)
        assert (
            elliptical.deflections_from_grid(grid=grid, bypass_decorator=True)
            == spherical.deflections_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = al.mass_profiles.EllipticalSersicRadialGradient(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = al.mass_profiles.SphericalSersicRadialGradient(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = al.Mask(mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        sersic = al.mass_profiles.EllipticalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        sersic = al.mass_profiles.SphericalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)
