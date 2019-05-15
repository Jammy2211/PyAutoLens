import numpy as np
import pytest

from autolens import dimensions as dim
from autolens.model.profiles import light_and_mass_profiles as lmp, light_profiles as lp, mass_profiles as mp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestSersic(object):

    def test__constructor_and_units(self):

        sersic = lmp.EllipticalSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=10.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = lmp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                    mass_to_light_ratio=10.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_sersic(self):
        sersic_lp = lp.EllipticalSersic(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                        sersic_index=2.0)
        sersic_mp = mp.EllipticalSersic(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                        sersic_index=2.0, mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalSersic(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                          sersic_index=2.0, mass_to_light_ratio=2.0)

        assert (sersic_lp.intensities_from_grid(grid) == sersic_lmp.intensities_from_grid(grid)).all()
        assert (sersic_mp.convergence_from_grid(grid) == sersic_lmp.convergence_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = lmp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                          effective_radius=1.0, sersic_index=2.0, mass_to_light_ratio=2.0)
        spherical = lmp.SphericalSersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.0,
                                        mass_to_light_ratio=2.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()
        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert (elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)).all()
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))


class TestExponential(object):

    def test__constructor_and_units(self):

        exponential = lmp.EllipticalExponential(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                               effective_radius=0.6, mass_to_light_ratio=10.0)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == 'angular / eps'

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = lmp.SphericalExponential(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6,
                                              mass_to_light_ratio=10.0)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == 'angular / eps'

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_exponential(self):
        sersic_lp = lp.EllipticalExponential(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6)
        sersic_mp = mp.EllipticalExponential(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                             mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalExponential(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                               mass_to_light_ratio=2.0)

        assert (sersic_lp.intensities_from_grid(grid) == sersic_lmp.intensities_from_grid(grid)).all()
        assert (sersic_mp.convergence_from_grid(grid) == sersic_lmp.convergence_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = lmp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                               effective_radius=1.0)
        spherical = lmp.SphericalExponential(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()
        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))


class TestDevVaucouleurs(object):

    def test__constructor_and_units(self):

        dev_vaucouleurs = lmp.EllipticalDevVaucouleurs(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                                      effective_radius=0.6, mass_to_light_ratio=10.0)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == 'angular / eps'

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = lmp.SphericalDevVaucouleurs(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6,
                                                     mass_to_light_ratio=10.0)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == 'angular / eps'

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_dev_vaucouleurs(self):
        sersic_lp = lp.EllipticalDevVaucouleurs(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6)
        sersic_mp = mp.EllipticalDevVaucouleurs(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalDevVaucouleurs(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                  mass_to_light_ratio=2.0)

        assert (sersic_lp.intensities_from_grid(grid) == sersic_lmp.intensities_from_grid(grid)).all()
        assert (sersic_mp.convergence_from_grid(grid) == sersic_lmp.convergence_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = lmp.EllipticalDevVaucouleurs(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                  effective_radius=1.0)
        spherical = lmp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()
        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))


class TestSersicRadialGradient(object):

    def test__constructor_and_units(self):

        sersic = lmp.EllipticalSersicRadialGradient(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=10.0,
                                                   mass_to_light_gradient=-1.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = lmp.SphericalSersicRadialGradient(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                    mass_to_light_ratio=10.0, mass_to_light_gradient=-1.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_sersic_radial_gradient(self):
        sersic_lp = lp.EllipticalSersic(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                        sersic_index=2.0)
        sersic_mp = mp.EllipticalSersicRadialGradient(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                      sersic_index=2.0, mass_to_light_ratio=2.0,
                                                      mass_to_light_gradient=0.5)
        sersic_lmp = lmp.EllipticalSersicRadialGradient(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=2.0, mass_to_light_ratio=2.0,
                                                        mass_to_light_gradient=0.5)

        assert (sersic_lp.intensities_from_grid(grid) == sersic_lmp.intensities_from_grid(grid)).all()
        assert (sersic_mp.convergence_from_grid(grid) == sersic_lmp.convergence_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = lmp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                        effective_radius=1.0)
        spherical = lmp.SphericalSersicRadialGradient(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()
        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))
