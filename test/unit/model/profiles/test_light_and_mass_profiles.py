import numpy as np

from autolens.model.profiles import light_and_mass_profiles as lmp, light_profiles as lp, mass_profiles as mp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestSersic(object):

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
