from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_and_mass_profiles as lmp
import pytest
import numpy as np

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

class TestSersic(object):

    def test__grid_calculations__same_as_sersic(self):

        sersic_lp = lp.EllipticalSersicLP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=2.0)
        sersic_mp = mp.EllipticalSersicMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=2.0, mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalSersicLMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=2.0, mass_to_light_ratio=2.0)

        assert (sersic_lp.intensity_from_grid(grid) == sersic_lmp.intensity_from_grid(grid)).all()
        assert (sersic_mp.surface_density_from_grid(grid) == sersic_lmp.surface_density_from_grid(grid)).all()
    #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):

        elliptical = lmp.EllipticalSersicLMP(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                             effective_radius=1.0, sersic_index=2.0, mass_to_light_ratio=2.0)
        spherical = lmp.SphericalSersicLMP(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.0,
                                           mass_to_light_ratio=2.0)

        assert (elliptical.intensity_from_grid(grid) == spherical.intensity_from_grid(grid)).all()
        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert (elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)).all()
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()


class TestExponential(object):

    def test__grid_calculations__same_as_exponential(self):
        
        sersic_lp = lp.EllipticalExponentialLP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6)
        sersic_mp = mp.EllipticalExponentialMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                               mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalExponentialLMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                  mass_to_light_ratio=2.0)

        assert (sersic_lp.intensity_from_grid(grid) == sersic_lmp.intensity_from_grid(grid)).all()
        assert (sersic_mp.surface_density_from_grid(grid) == sersic_lmp.surface_density_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):

        elliptical = lmp.EllipticalExponentialLMP(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                  effective_radius=1.0)
        spherical = lmp.SphericalExponentialLMP(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensity_from_grid(grid) == spherical.intensity_from_grid(grid)).all()
        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()

class TestDevVaucouleurs(object):

    def test__grid_calculations__same_as_dev_vaucouleurs(self):
        
        sersic_lp = lp.EllipticalDevVaucouleursLP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6)
        sersic_mp = mp.EllipticalDevVaucouleursMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                  mass_to_light_ratio=2.0)
        sersic_lmp = lmp.EllipticalDevVaucouleursLMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                                     mass_to_light_ratio=2.0)

        assert (sersic_lp.intensity_from_grid(grid) == sersic_lmp.intensity_from_grid(grid)).all()
        assert (sersic_mp.surface_density_from_grid(grid) == sersic_lmp.surface_density_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):

        elliptical = lmp.EllipticalDevVaucouleursLMP(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                     effective_radius=1.0)
        spherical = lmp.SphericalDevVaucouleursLMP(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensity_from_grid(grid) == spherical.intensity_from_grid(grid)).all()
        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()


class TestSersicRadialGradient(object):

    def test__grid_calculations__same_as_sersic_radial_gradient(self):

        sersic_lp = lp.EllipticalSersicLP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                          sersic_index=2.0)
        sersic_mp = mp.EllipticalSersicRadialGradientMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                          sersic_index=2.0, mass_to_light_ratio=2.0, mass_to_light_gradient=0.5)
        sersic_lmp = lmp.EllipticalSersicRadialGradientLMP(axis_ratio=0.7, phi=1.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=2.0, mass_to_light_ratio=2.0, mass_to_light_gradient=0.5)

        assert (sersic_lp.intensity_from_grid(grid) == sersic_lmp.intensity_from_grid(grid)).all()
        assert (sersic_mp.surface_density_from_grid(grid) == sersic_lmp.surface_density_from_grid(grid)).all()
        #    assert (sersic_mp.potential_from_grid(grid) == sersic_lmp.potential_from_grid(grid)).all()
        assert (sersic_mp.deflections_from_grid(grid) == sersic_lmp.deflections_from_grid(grid)).all()

    def test__spherical_and_elliptical_identical(self):

        elliptical = lmp.EllipticalSersicRadialGradientLMP(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                           effective_radius=1.0)
        spherical = lmp.SphericalSersicRadialGradientLMP(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0)

        assert (elliptical.intensity_from_grid(grid) == spherical.intensity_from_grid(grid)).all()
        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()