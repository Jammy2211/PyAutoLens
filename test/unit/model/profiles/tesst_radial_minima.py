from __future__ import division, print_function

import autofit as af
import os

test_path = "{}/config/".format(os.path.dirname(os.path.realpath(__file__)))
af.conf.instance = af.conf.Config(config_path=test_path)

import numpy as np
import pytest

from autolens.array import mask as msk
from autolens.array import grids
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp


class TestGaussian:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        from autolens.model.profiles import light_profiles as lp

        gaussian = al.EllipticalGaussian(centre=(0.0, 0.0))

        image_1 = gaussian.profile_image_from_grid(grid=grid_01)
        image_0 = gaussian.profile_image_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        gaussian = al.EllipticalGaussian(centre=(1.0, 1.0))

        image_1 = gaussian.profile_image_from_grid(grid=grid_22)
        image_0 = gaussian.profile_image_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        gaussian = al.SphericalGaussian(centre=(0.0, 0.0))

        image_1 = gaussian.profile_image_from_grid(grid=grid_01)
        image_0 = gaussian.profile_image_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        gaussian = al.SphericalGaussian(centre=(1.0, 1.0))

        image_1 = gaussian.profile_image_from_grid(grid=grid_22)
        image_0 = gaussian.profile_image_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestSersic:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        sersic = al.EllipticalSersic(centre=(0.0, 0.0))

        image_1 = sersic.profile_image_from_grid(grid=grid_01)
        image_0 = sersic.profile_image_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        sersic = al.EllipticalSersic(centre=(1.0, 1.0))

        image_1 = sersic.profile_image_from_grid(grid=grid_22)
        image_0 = sersic.profile_image_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        sersic = al.light_profiles.SphericalSersic(centre=(0.0, 0.0))

        image_1 = sersic.profile_image_from_grid(grid=grid_01)
        image_0 = sersic.profile_image_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        sersic = al.light_profiles.SphericalSersic(centre=(1.0, 1.0))

        image_1 = sersic.profile_image_from_grid(grid=grid_22)
        image_0 = sersic.profile_image_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestExponential:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        exponential = al.EllipticalExponential(centre=(0.0, 0.0))

        image_1 = exponential.profile_image_from_grid(grid=grid_01)
        image_0 = exponential.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        exponential = al.EllipticalExponential(centre=(1.0, 1.0))

        image_1 = exponential.profile_image_from_grid(grid=grid_22)
        image_0 = exponential.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(0.0, 0.0))

        image_1 = exponential.profile_image_from_grid(grid=grid_01)
        image_0 = exponential.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(1.0, 1.0))

        image_1 = exponential.profile_image_from_grid(grid=grid_22)
        image_0 = exponential.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestDevVaucouleurs:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        image_1 = dev_vaucouleurs.profile_image_from_grid(grid=grid_01)
        image_0 = dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(1.0, 1.0))

        image_1 = dev_vaucouleurs.profile_image_from_grid(grid=grid_22)
        image_0 = dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(0.0, 0.0))

        image_1 = dev_vaucouleurs.profile_image_from_grid(grid=grid_01)
        image_0 = dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(1.0, 1.0))

        image_1 = dev_vaucouleurs.profile_image_from_grid(grid=grid_22)
        image_0 = dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestCoreSersic(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        core_sersic = al.EllipticalCoreSersic(centre=(0.0, 0.0))

        image_1 = core_sersic.profile_image_from_grid(grid=grid_01)
        image_0 = core_sersic.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        core_sersic = al.EllipticalCoreSersic(centre=(1.0, 1.0))

        image_1 = core_sersic.profile_image_from_grid(grid=grid_22)
        image_0 = core_sersic.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        core_sersic = al.SphericalCoreSersic(centre=(0.0, 0.0))

        image_1 = core_sersic.profile_image_from_grid(grid=grid_01)
        image_0 = core_sersic.profile_image_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        core_sersic = al.SphericalCoreSersic(centre=(1.0, 1.0))

        image_1 = core_sersic.profile_image_from_grid(grid=grid_22)
        image_0 = core_sersic.profile_image_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestPointMass(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        point_mass = al.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections_1 = point_mass.deflections_from_grid(grid=grid_01)
        deflections_0 = point_mass.deflections_from_grid(
            grid=np.array([[0.00000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        point_mass = al.PointMass(centre=(1.0, 1.0), einstein_radius=1.0)

        deflections_1 = point_mass.deflections_from_grid(grid=grid_22)
        deflections_0 = point_mass.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestCoredPowerLaw(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        cored_power_law = al.EllipticalCoredPowerLaw(
            centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0
        )

        convergence_1 = cored_power_law.convergence_from_grid(grid=grid_01)
        convergence_0 = cored_power_law.convergence_from_grid(
            grid=np.array([[1e-8, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)


class TestPowerLaw(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        power_law = al.EllipticalPowerLaw(
            centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0
        )

        convergence_1 = power_law.convergence_from_grid(grid=grid_01)
        convergence_0 = power_law.convergence_from_grid(grid=np.array([[1.0e-9, 0.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        power_law = al.SphericalPowerLaw(
            centre=(-0.7, 0.5), einstein_radius=1.3, slope=1.8
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

        mask = al.Mask(mask, pixel_scale=1.0)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = power_law.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = power_law.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()


class TestCoredIsothermal(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_01)
        convergence_0 = cored_isothermal.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_22)
        convergence_0 = cored_isothermal.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_01)
        convergence_0 = cored_isothermal.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_22)
        convergence_0 = cored_isothermal.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_01)
        potential_0 = cored_isothermal.potential_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_22)
        potential_0 = cored_isothermal.potential_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_01)
        potential_0 = cored_isothermal.potential_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_22)
        potential_0 = cored_isothermal.potential_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_01)
        deflections_0 = cored_isothermal.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        cored_isothermal = al.EllipticalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_22)
        deflections_0 = cored_isothermal.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_01)
        deflections_0 = cored_isothermal.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        cored_isothermal = al.SphericalCoredIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_22)
        deflections_0 = cored_isothermal.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestIsothermal(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = isothermal.convergence_from_grid(grid=grid_01)
        convergence_0 = isothermal.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        convergence_1 = isothermal.convergence_from_grid(grid=grid_22)
        convergence_0 = isothermal.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = isothermal.convergence_from_grid(grid=grid_01)
        convergence_0 = isothermal.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        convergence_1 = isothermal.convergence_from_grid(grid=grid_22)
        convergence_0 = isothermal.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = isothermal.potential_from_grid(grid=grid_01)
        potential_0 = isothermal.potential_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        potential_1 = isothermal.potential_from_grid(grid=grid_22)
        potential_0 = isothermal.potential_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = isothermal.potential_from_grid(grid=grid_01)
        potential_0 = isothermal.potential_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        potential_1 = isothermal.potential_from_grid(grid=grid_22)
        potential_0 = isothermal.potential_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = isothermal.deflections_from_grid(grid=grid_01)
        deflections_0 = isothermal.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        isothermal = al.mass_profiles.EllipticalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        deflections_1 = isothermal.deflections_from_grid(grid=grid_22)
        deflections_0 = isothermal.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = isothermal.deflections_from_grid(grid=grid_01)
        deflections_0 = isothermal.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        deflections_1 = isothermal.deflections_from_grid(grid=grid_22)
        deflections_0 = isothermal.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestGeneralizedNFW(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        convergence_1 = gnfw.convergence_from_grid(grid=grid_01)
        convergence_0 = gnfw.convergence_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0))

        convergence_1 = gnfw.convergence_from_grid(grid=grid_22)
        convergence_0 = gnfw.convergence_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        potential_1 = gnfw.potential_from_grid(grid=grid_01)
        potential_0 = gnfw.potential_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0))

        potential_1 = gnfw.potential_from_grid(grid=grid_22)
        potential_0 = gnfw.potential_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        deflections_1 = gnfw.deflections_from_grid(grid=grid_01)
        deflections_0 = gnfw.deflections_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        gnfw = al.mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0))

        deflections_1 = gnfw.deflections_from_grid(grid=grid_22)
        deflections_0 = gnfw.deflections_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestTruncatedNFW(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(0.0, 0.0))

        convergence_1 = truncated_nfw.convergence_from_grid(grid=grid_01)
        convergence_0 = truncated_nfw.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(1.0, 1.0))

        convergence_1 = truncated_nfw.convergence_from_grid(grid=grid_22)
        convergence_0 = truncated_nfw.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(0.0, 0.0))

        potential_1 = truncated_nfw.potential_from_grid(grid=grid_01)
        potential_0 = truncated_nfw.potential_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(1.0, 1.0))

        potential_1 = truncated_nfw.potential_from_grid(grid=grid_22)
        potential_0 = truncated_nfw.potential_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(0.0, 0.0))

        deflections_1 = truncated_nfw.deflections_from_grid(grid=grid_01)
        deflections_0 = truncated_nfw.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        truncated_nfw = al.mass_profiles.SphericalTruncatedNFW(centre=(1.0, 1.0))

        deflections_1 = truncated_nfw.deflections_from_grid(grid=grid_22)
        deflections_0 = truncated_nfw.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestNFW(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        nfw = al.mass_profiles.EllipticalNFW(centre=(0.0, 0.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_01)
        convergence_0 = nfw.convergence_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = al.mass_profiles.EllipticalNFW(centre=(1.0, 1.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_22)
        convergence_0 = nfw.convergence_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(0.0, 0.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_01)
        convergence_0 = nfw.convergence_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(1.0, 1.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_22)
        convergence_0 = nfw.convergence_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = al.mass_profiles.EllipticalNFW(centre=(0.0, 0.0))

        potential_1 = nfw.potential_from_grid(grid=grid_01)
        potential_0 = nfw.potential_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = al.mass_profiles.EllipticalNFW(centre=(1.0, 1.0))

        potential_1 = nfw.potential_from_grid(grid=grid_22)
        potential_0 = nfw.potential_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(0.0, 0.0))

        potential_1 = nfw.potential_from_grid(grid=grid_01)
        potential_0 = nfw.potential_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(1.0, 1.0))

        potential_1 = nfw.potential_from_grid(grid=grid_22)
        potential_0 = nfw.potential_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = al.mass_profiles.EllipticalNFW(centre=(0.0, 0.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_01)
        deflections_0 = nfw.deflections_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        nfw = al.mass_profiles.EllipticalNFW(centre=(1.0, 1.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_22)
        deflections_0 = nfw.deflections_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(0.0, 0.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_01)
        deflections_0 = nfw.deflections_from_grid(grid=np.array([[0.000000001, 0.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        nfw = al.mass_profiles.SphericalNFW(centre=(1.0, 1.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_22)
        deflections_0 = nfw.deflections_from_grid(grid=np.array([[1.000000001, 1.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestSersic(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        sersic = al.EllipticalSersic(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_01)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.EllipticalSersic(centre=(1.0, 1.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_22)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.SphericalSersic(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_01)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.SphericalSersic(centre=(1.0, 1.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_22)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.EllipticalSersic(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_01)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.EllipticalSersic(centre=(1.0, 1.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_22)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.SphericalSersic(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_01)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.SphericalSersic(centre=(1.0, 1.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_22)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestExponential(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        exponential = al.EllipticalExponential(centre=(0.0, 0.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_01)
        convergence_0 = exponential.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = al.EllipticalExponential(centre=(1.0, 1.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_22)
        convergence_0 = exponential.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(0.0, 0.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_01)
        convergence_0 = exponential.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(1.0, 1.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_22)
        convergence_0 = exponential.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = al.EllipticalExponential(centre=(0.0, 0.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_01)
        deflections_0 = exponential.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        exponential = al.EllipticalExponential(centre=(1.0, 1.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_22)
        deflections_0 = exponential.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(0.0, 0.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_01)
        deflections_0 = exponential.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        exponential = al.SphericalExponential(centre=(1.0, 1.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_22)
        deflections_0 = exponential.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestDevVaucouleurs(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        convergence_1 = dev_vaucouleurs.convergence_from_grid(grid=grid_01)
        convergence_0 = dev_vaucouleurs.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(1.0, 1.0))

        convergence_1 = dev_vaucouleurs.convergence_from_grid(grid=grid_22)
        convergence_0 = dev_vaucouleurs.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(0.0, 0.0))

        convergence_1 = dev_vaucouleurs.convergence_from_grid(grid=grid_01)
        convergence_0 = dev_vaucouleurs.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(1.0, 1.0))

        convergence_1 = dev_vaucouleurs.convergence_from_grid(grid=grid_22)
        convergence_0 = dev_vaucouleurs.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_01)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        dev_vaucouleurs = al.EllipticalDevVaucouleurs(centre=(1.0, 1.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_22)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(0.0, 0.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_01)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        dev_vaucouleurs = al.SphericalDevVaucouleurs(centre=(1.0, 1.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_22)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestSersicMassRadialGradient(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        sersic = al.EllipticalSersicRadialGradient(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_01)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.EllipticalSersicRadialGradient(centre=(1.0, 1.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_22)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.SphericalSersicRadialGradient(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_01)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.SphericalSersicRadialGradient(centre=(1.0, 1.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_22)
        convergence_0 = sersic.convergence_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = al.EllipticalSersicRadialGradient(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_01)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.EllipticalSersicRadialGradient(centre=(1.0, 1.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_22)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.SphericalSersicRadialGradient(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_01)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = al.SphericalSersicRadialGradient(centre=(1.0, 1.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_22)
        deflections_0 = sersic.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)
        sersic = al.SphericalSersicRadialGradient(
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

        mask = al.Mask(mask, pixel_scale=1.0)

        grid = al.Grid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()


class TestMassSheet(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        mass_sheet = al.mass_profiles.MassSheet(centre=(0.0, 0.0))

        deflections_1 = mass_sheet.deflections_from_grid(grid=grid_01)
        deflections_0 = mass_sheet.deflections_from_grid(
            grid=np.array([[0.000000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        mass_sheet = al.mass_profiles.MassSheet(centre=(1.0, 1.0))

        deflections_1 = mass_sheet.deflections_from_grid(grid=grid_22)
        deflections_0 = mass_sheet.deflections_from_grid(
            grid=np.array([[1.000000001, 1.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestExternalShear(object):
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):
        shear = al.mass_profiles.ExternalShear(magnitude=0.1, phi=45.0)

        deflections_1 = shear.deflections_from_grid(grid=np.array([[1e-8, 0.0]]))
        deflections_0 = shear.deflections_from_grid(grid=np.array([[1e-9, 0.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)
