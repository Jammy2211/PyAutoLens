import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import exc, dimensions as dim
from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.lens import plane as pl
from autolens.lens.util import lens_util
from autolens.model import cosmology_util
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util
from autolens.model.inversion import pixelizations, regularization
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.unit.mock.data import mock_ccd, mock_grids
from test.unit.mock.model import mock_inversion as mock_inv
from test.unit.mock.model.mock_cosmology import MockCosmology

planck = cosmo.Planck15


class TestAbstractPlane(object):

    class TestCosmology:

        def test__all_cosmological_quantities_match_cosmology_util(self):

            plane = pl.AbstractPlane(redshift=0.1, cosmology=planck)

            assert plane.arcsec_per_kpc == cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
                redshift=0.1, cosmology=planck)

            assert plane.kpc_per_arcsec == \
                   cosmology_util.kpc_per_arcsec_from_redshift_and_cosmology(redshift=0.1, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='arcsec') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=0.1, cosmology=planck, unit_length='arcsec')

            plane = pl.AbstractPlane(redshift=0.1, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=0.1, cosmology=planck, unit_length='kpc')

            plane = pl.AbstractPlane(redshift=1.0, cosmology=planck)

            assert plane.arcsec_per_kpc == cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
                redshift=1.0, cosmology=planck)

            assert plane.kpc_per_arcsec == \
                   cosmology_util.kpc_per_arcsec_from_redshift_and_cosmology(redshift=1.0, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='arcsec') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=1.0, cosmology=planck, unit_length='arcsec')

            plane = pl.AbstractPlane(redshift=1.0, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=1.0, cosmology=planck, unit_length='kpc')

            plane = pl.AbstractPlane(redshift=0.6)

            assert plane.cosmic_average_density_in_units(unit_length='arcsec', unit_mass='solMass') == \
                   cosmology_util.cosmic_average_density_from_redshift_and_cosmology(
                       redshift=0.6, cosmology=planck, unit_length='arcsec', unit_mass='solMass')

            plane = pl.AbstractPlane(redshift=0.6, cosmology=planck)

            assert plane.cosmic_average_density_in_units(unit_length='kpc', unit_mass='solMass') == \
                   cosmology_util.cosmic_average_density_from_redshift_and_cosmology(
                       redshift=0.6, cosmology=planck, unit_length='kpc', unit_mass='solMass')

    class TestProperties:

        def test__has_light_profile(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_light_profile is False

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())], redshift=None)
            assert plane.has_light_profile is True

            plane = pl.AbstractPlane(
                galaxies=[g.Galaxy(redshift=0.5, light_profile=lp.LightProfile()), g.Galaxy(redshift=0.5)],
                redshift=None)
            assert plane.has_light_profile is True

        def test__has_mass_profile(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_mass_profile is False

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5, mass_profile=mp.MassProfile())], redshift=None)
            assert plane.has_mass_profile is True

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5, mass_profile=mp.MassProfile()),
                                               g.Galaxy(redshift=0.5)],
                                     redshift=None)
            assert plane.has_mass_profile is True

        def test__has_pixelization(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_pixelization is False

            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_pixelization is True

            plane = pl.AbstractPlane(galaxies=[galaxy_pix, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_pixelization is True

        def test__has_regularization(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_regularization is False

            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_regularization is True

            plane = pl.AbstractPlane(galaxies=[galaxy_pix, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_regularization is True

        def test__has_hyper_galaxy(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_hyper_galaxy is False

            galaxy = g.Galaxy(redshift=0.5, hyper_galaxy=g.HyperGalaxy())

            plane = pl.AbstractPlane(galaxies=[galaxy], redshift=None)
            assert plane.has_hyper_galaxy is True

            plane = pl.AbstractPlane(galaxies=[galaxy, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_hyper_galaxy is True

    class TestRegularization:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self):
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            plane = pl.AbstractPlane(galaxies=[galaxy_no_pix], redshift=None)

            assert plane.regularization is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self):
            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                  regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)

            assert plane.regularization.shape == (1, 1)

            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                  regularization=mock_inv.MockRegularization(matrix_shape=(2, 2)))
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            plane = pl.AbstractPlane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=None)

            assert plane.regularization.shape == (2, 2)

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self):
            galaxy_pix_0 = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                    regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))
            galaxy_pix_1 = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=2),
                                    regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))

            plane = pl.AbstractPlane(galaxies=[galaxy_pix_0, galaxy_pix_1], redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.regularization)

    class TestLuminosities:

        def test__within_circle_different_luminosity_units__same_as_galaxy_luminosities(self):
            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius,
                                                                                        unit_luminosity='eps')

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts',
                                                                 exposure_time=3.0)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts',
                                                                 exposure_time=3.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius,
                                                                                        unit_luminosity='counts',
                                                                                        exposure_time=3.0)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_circle_different_distance_units__same_as_galaxy_luminosities(self):
            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            radius = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius)

            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_ellipse_different_luminosity_units__same_as_galaxy_luminosities(self):
            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='eps')
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='eps')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                                         unit_luminosity='eps')

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='counts',
                                                                  exposure_time=3.0)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='counts',
                                                                  exposure_time=3.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                                         unit_luminosity='counts',
                                                                                         exposure_time=3.0)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_ellipse_different_distance_units__same_as_galaxy_luminosities(self):
            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            major_axis = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis)

            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

    class TestMasses:

        def test__within_circle_different_mass_units__same_as_galaxy_masses(self):
            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_circle_in_units(radius=radius, unit_mass='angular')
            g1_mass = g1.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius, unit_mass='angular')

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            g0_mass = g0.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                     redshift_source=1.0)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                     redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(
                radius=radius, unit_mass='solMass', redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_circle_different_distance_units__same_as_galaxy_masses(self):
            radius = dim.Length(1.0, 'arcsec')

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_mass = g0.mass_within_circle_in_units(radius=radius, redshift_source=1.0)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius,
                                                                            redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            radius = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_mass = g0.mass_within_circle_in_units(radius=radius, redshift_source=1.0,
                                                     kpc_per_arcsec=plane.kpc_per_arcsec)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, redshift_source=1.0,
                                                     kpc_per_arcsec=plane.kpc_per_arcsec)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius,
                                                                            redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_ellipse_different_mass_units__same_as_galaxy_masses(self):
            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='angular')
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='angular')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             unit_mass='angular')

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='solMass',
                                                      redshift_source=1.0)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='solMass',
                                                      redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             unit_mass='solMass',
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_ellipse_different_distance_units__same_as_galaxy_masses(self):
            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis,
                                                      redshift_source=1.0)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis,
                                                      redshift_source=1.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            major_axis = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, redshift_source=1.0,
                                                      kpc_per_arcsec=plane.kpc_per_arcsec)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, redshift_source=1.0,
                                                      kpc_per_arcsec=plane.kpc_per_arcsec)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

    class TestEinsteinRadiiAndMass:

        def test__plane_has_galaxies_with_sis_profiles__einstein_radius_and_mass_sum_of_sis_profiles(self):
            cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0)

            sis_0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            sis_1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            plane = pl.AbstractPlane(galaxies=[sis_0], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(1.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) \
                   == pytest.approx(2.0 * np.pi, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_1], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == pytest.approx(4.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi * 2.0 ** 2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) == \
                   pytest.approx(2.0 * np.pi * 2.0 ** 2.0, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_0, sis_1], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(3.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == \
                   pytest.approx(2.0 * 3.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi * (1.0 + 2.0 ** 2.0),
                                                                                      1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) == \
                   pytest.approx(2.0 * np.pi * (1.0 + 2.0 ** 2.0), 1.0e-4)

        def test__include_galaxy_with_no_mass_profile__does_not_impact_einstein_radius_or_mass(self):
            sis_0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            sis_1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))
            g0 = g.Galaxy(redshift=0.5)

            plane = pl.AbstractPlane(galaxies=[sis_0, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(1.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_1, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi * 2.0 ** 2.0, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_0, sis_1, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(3.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi * (1.0 + 2.0 ** 2.0),
                                                                                      1.0e-4)

        def test__only_galaxies_without_mass_profiles__einstein_radius_and_mass_are_none(self):
            g0 = g.Galaxy(redshift=0.5)

            plane = pl.AbstractPlane(galaxies=[g0], redshift=0.5)

            assert plane.einstein_radius_in_units() is None
            assert plane.einstein_mass_in_units() is None

            plane = pl.AbstractPlane(galaxies=[g0, g0], redshift=0.5)

            assert plane.einstein_radius_in_units() is None
            assert plane.einstein_mass_in_units() is None

    class TestMassProfileGeometry:

        def test__extract_centres_of_all_mass_profiles_of_all_galaxies(self):

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(centre=(1.0, 1.0)))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(centre=(2.0, 2.0)))
            g2 = g.Galaxy(redshift=0.5, mass0=mp.SphericalIsothermal(centre=(3.0, 3.0)),
                          mass1=mp.SphericalIsothermal(centre=(4.0, 4.0)))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == []

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(2.0, 2.0)], [(1.0, 1.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5), g2],
                                     redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)], [(3.0, 3.0), (4.0, 4.0)]]

        def test__extracts_axis_ratio_of_all_mass_profiles_of_all_galaxies(self):
            g0 = g.Galaxy(redshift=0.5, mass=mp.EllipticalIsothermal(axis_ratio=0.9))
            g1 = g.Galaxy(redshift=0.5, mass=mp.EllipticalIsothermal(axis_ratio=0.8))
            g2 = g.Galaxy(redshift=0.5, mass0=mp.EllipticalIsothermal(axis_ratio=0.7),
                          mass1=mp.EllipticalIsothermal(axis_ratio=0.6))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == []

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.8], [0.9]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5), g2],
                                     redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8], [0.7, 0.6]]

        def test__extracts_phi_of_all_mass_profiles_of_all_galaxies(self):
            g0 = g.Galaxy(redshift=0.5, mass=mp.EllipticalIsothermal(phi=0.9))
            g1 = g.Galaxy(redshift=0.5, mass=mp.EllipticalIsothermal(phi=0.8))
            g2 = g.Galaxy(redshift=0.5, mass0=mp.EllipticalIsothermal(phi=0.7),
                          mass1=mp.EllipticalIsothermal(phi=0.6))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == []

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.8], [0.9]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5)], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(redshift=0.5), g1, g.Galaxy(redshift=0.5), g2],
                                     redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8], [0.7, 0.6]]

    class TestSummarize:

        def test__plane_x2_galaxies__summarize_is_correct(self):
            sersic_0 = lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)
            sersic_1 = lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)

            sis_0 = mp.SphericalIsothermal(einstein_radius=1.0)
            sis_1 = mp.SphericalIsothermal(einstein_radius=2.0)

            g0 = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1,
                          mass_profile_0=sis_0, mass_profile_1=sis_1)

            g1 = g.Galaxy(redshift=0.6, light_profile_0=sersic_0, mass_profile_0=sis_0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.6)

            summary_text = plane.summarize_in_units(radii=[dim.Length(10.0), dim.Length(500.0)], whitespace=50,
                                                    unit_length='arcsec', unit_luminosity='eps', unit_mass='angular')

            i = 0
            assert summary_text[i] == 'Plane\n';
            i += 1
            assert summary_text[i] == 'redshift                                          0.60';
            i += 1
            assert summary_text[i] == 'kpc_per_arcsec                                    6.88';
            i += 1
            assert summary_text[i] == 'angular_diameter_distance_to_earth                206264.81';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == 'Galaxy\n';
            i += 1
            assert summary_text[i] == 'redshift                                          0.50';
            i += 1
            assert summary_text[i] == '\nGALAXY LIGHT\n\n';
            i += 1
            assert summary_text[i] == 'luminosity_within_10.00_arcsec                    1.8854e+02 eps';
            i += 1
            assert summary_text[i] == 'luminosity_within_500.00_arcsec                   1.9573e+02 eps';
            i += 1
            assert summary_text[i] == '\nLIGHT PROFILES:\n\n';
            i += 1
            assert summary_text[i] == 'Light Profile = SphericalSersic\n';
            i += 1
            assert summary_text[i] == 'luminosity_within_10.00_arcsec                    6.2848e+01 eps';
            i += 1
            assert summary_text[i] == 'luminosity_within_500.00_arcsec                   6.5243e+01 eps';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == 'Light Profile = SphericalSersic\n';
            i += 1
            assert summary_text[i] == 'luminosity_within_10.00_arcsec                    1.2570e+02 eps';
            i += 1
            assert summary_text[i] == 'luminosity_within_500.00_arcsec                   1.3049e+02 eps';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == '\nGALAXY MASS\n\n';
            i += 1
            assert summary_text[i] == 'einstein_radius                                   3.00 arcsec';
            i += 1
            assert summary_text[i] == 'einstein_mass                                     1.5708e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_10.00_arcsec                          9.4248e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_500.00_arcsec                         4.7124e+03 angular';
            i += 1
            assert summary_text[i] == '\nMASS PROFILES:\n\n';
            i += 1
            assert summary_text[i] == 'Mass Profile = SphericalIsothermal\n';
            i += 1
            assert summary_text[i] == 'einstein_radius                                   1.00 arcsec';
            i += 1
            assert summary_text[i] == 'einstein_mass                                     3.1416e+00 angular';
            i += 1
            assert summary_text[i] == 'mass_within_10.00_arcsec                          3.1416e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_500.00_arcsec                         1.5708e+03 angular';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == 'Mass Profile = SphericalIsothermal\n';
            i += 1
            assert summary_text[i] == 'einstein_radius                                   2.00 arcsec';
            i += 1
            assert summary_text[i] == 'einstein_mass                                     1.2566e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_10.00_arcsec                          6.2832e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_500.00_arcsec                         3.1416e+03 angular';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == 'Galaxy\n';
            i += 1
            assert summary_text[i] == 'redshift                                          0.60';
            i += 1
            assert summary_text[i] == '\nGALAXY LIGHT\n\n';
            i += 1
            assert summary_text[i] == 'luminosity_within_10.00_arcsec                    6.2848e+01 eps';
            i += 1
            assert summary_text[i] == 'luminosity_within_500.00_arcsec                   6.5243e+01 eps';
            i += 1
            assert summary_text[i] == '\nLIGHT PROFILES:\n\n';
            i += 1
            assert summary_text[i] == 'Light Profile = SphericalSersic\n';
            i += 1
            assert summary_text[i] == 'luminosity_within_10.00_arcsec                    6.2848e+01 eps';
            i += 1
            assert summary_text[i] == 'luminosity_within_500.00_arcsec                   6.5243e+01 eps';
            i += 1
            assert summary_text[i] == '\n';
            i += 1
            assert summary_text[i] == '\nGALAXY MASS\n\n';
            i += 1
            assert summary_text[i] == 'einstein_radius                                   1.00 arcsec';
            i += 1
            assert summary_text[i] == 'einstein_mass                                     3.1416e+00 angular';
            i += 1
            assert summary_text[i] == 'mass_within_10.00_arcsec                          3.1416e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_500.00_arcsec                         1.5708e+03 angular';
            i += 1
            assert summary_text[i] == '\nMASS PROFILES:\n\n';
            i += 1
            assert summary_text[i] == 'Mass Profile = SphericalIsothermal\n';
            i += 1
            assert summary_text[i] == 'einstein_radius                                   1.00 arcsec';
            i += 1
            assert summary_text[i] == 'einstein_mass                                     3.1416e+00 angular';
            i += 1
            assert summary_text[i] == 'mass_within_10.00_arcsec                          3.1416e+01 angular';
            i += 1
            assert summary_text[i] == 'mass_within_500.00_arcsec                         1.5708e+03 angular';
            i += 1
            assert summary_text[i] == '\n';
            i += 1


class TestAbstractPlaneGridded(object):

    class TestImage:

        def test__image_from_plane__same_as_its_light_profile_image(self, grid_stack_5x5, gal_x1_lp):
            light_profile = gal_x1_lp.light_profiles[0]

            lp_sub_image = light_profile.intensities_from_grid(grid_stack_5x5.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_lp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_image_1d[0] == lp_image_pixel_0).all()
            assert (plane.profile_image_plane_image_1d[1] == lp_image_pixel_1).all()
            assert (plane.profile_image_plane_image_2d ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.profile_image_plane_image_1d)).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grid_stack_5x5, gal_x1_lp):
            galaxy_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.sub, galaxies=[gal_x1_lp])

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_lp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.profile_image_plane_image_1d == pytest.approx(galaxy_image, 1.0e-4)

            image_plane_image = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.profile_image_plane_image_1d)

            assert plane.profile_image_plane_image_2d == pytest.approx(image_plane_image, 1.0e-4)

        def test__single_multiple_intensity(self, grid_stack_5x5):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None,
                                            redshift=None)

            assert (plane.profile_image_plane_image_1d_of_galaxies[0] == plane.profile_image_plane_image_1d_of_galaxy(
                g0)).all()

        def test__image_plane_image_of_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_sub_image = lp0.intensities_from_grid(grid_stack_5x5.sub)
            lp1_sub_image = lp1.intensities_from_grid(grid_stack_5x5.sub)

            # Perform sub gridding average manually
            lp0_image_pixel_0 = (lp0_sub_image[0] + lp0_sub_image[1] + lp0_sub_image[2] + lp0_sub_image[3]) / 4
            lp0_image_pixel_1 = (lp0_sub_image[4] + lp0_sub_image[5] + lp0_sub_image[6] + lp0_sub_image[7]) / 4
            lp1_image_pixel_0 = (lp1_sub_image[0] + lp1_sub_image[1] + lp1_sub_image[2] + lp1_sub_image[3]) / 4
            lp1_image_pixel_1 = (lp1_sub_image[4] + lp1_sub_image[5] + lp1_sub_image[6] + lp1_sub_image[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.profile_image_plane_image_1d[0] == pytest.approx(lp0_image_pixel_0 + lp1_image_pixel_0, 1.0e-4)
            assert plane.profile_image_plane_image_1d[1] == pytest.approx(lp0_image_pixel_1 + lp1_image_pixel_1, 1.0e-4)

            image_plane_image = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.profile_image_plane_image_1d)

            assert plane.profile_image_plane_image_2d == image_plane_image

            assert plane.profile_image_plane_image_1d_of_galaxies[0][0] == lp0_image_pixel_0
            assert plane.profile_image_plane_image_1d_of_galaxies[0][1] == lp0_image_pixel_1
            assert plane.profile_image_plane_image_1d_of_galaxies[1][0] == lp1_image_pixel_0
            assert plane.profile_image_plane_image_1d_of_galaxies[1][1] == lp1_image_pixel_1

        def test__same_as_above__use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.sub, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.sub, galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.profile_image_plane_image_1d == pytest.approx(g0_image + g1_image, 1.0e-4)
            assert (plane.profile_image_plane_image_2d ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.profile_image_plane_image_1d)).all()

            assert (plane.profile_image_plane_image_1d_of_galaxies[0] == g0_image).all()
            assert (plane.profile_image_plane_image_1d_of_galaxies[1] == g1_image).all()

        def test__padded_grid_stack_in__profile_image_plane_image_is_padded(self, padded_grid_stack_5x5, gal_x1_lp):
            light_profile = gal_x1_lp.light_profiles[0]

            lp_sub_image = light_profile.intensities_from_grid(padded_grid_stack_5x5.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4
            lp_image_pixel_2 = (lp_sub_image[8] + lp_sub_image[9] + lp_sub_image[10] + lp_sub_image[11]) / 4
            lp_image_pixel_3 = (lp_sub_image[12] + lp_sub_image[13] + lp_sub_image[14] + lp_sub_image[15]) / 4
            lp_image_pixel_4 = (lp_sub_image[16] + lp_sub_image[17] + lp_sub_image[18] + lp_sub_image[19]) / 4
            lp_image_pixel_5 = (lp_sub_image[20] + lp_sub_image[21] + lp_sub_image[22] + lp_sub_image[23]) / 4
            lp_image_pixel_6 = (lp_sub_image[24] + lp_sub_image[25] + lp_sub_image[26] + lp_sub_image[27]) / 4
            lp_image_pixel_7 = (lp_sub_image[28] + lp_sub_image[29] + lp_sub_image[30] + lp_sub_image[31]) / 4
            lp_image_pixel_8 = (lp_sub_image[32] + lp_sub_image[33] + lp_sub_image[34] + lp_sub_image[35]) / 4
            lp_image_pixel_9 = (lp_sub_image[36] + lp_sub_image[37] + lp_sub_image[38] + lp_sub_image[39]) / 4
            lp_image_pixel_10 = (lp_sub_image[40] + lp_sub_image[41] + lp_sub_image[42] + lp_sub_image[43]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_lp], grid_stack=padded_grid_stack_5x5,
                                            compute_deflections=False, border=None, redshift=None)

            assert plane.profile_image_plane_image_2d_for_simulation.shape == (7, 7)
            assert plane.profile_image_plane_image_2d_for_simulation[0, 0] == lp_image_pixel_0
            assert plane.profile_image_plane_image_2d_for_simulation[0, 1] == lp_image_pixel_1
            assert plane.profile_image_plane_image_2d_for_simulation[0, 2] == lp_image_pixel_2
            assert plane.profile_image_plane_image_2d_for_simulation[0, 3] == lp_image_pixel_3
            assert plane.profile_image_plane_image_2d_for_simulation[0, 4] == lp_image_pixel_4
            assert plane.profile_image_plane_image_2d_for_simulation[0, 5] == lp_image_pixel_5
            assert plane.profile_image_plane_image_2d_for_simulation[0, 6] == lp_image_pixel_6
            assert plane.profile_image_plane_image_2d_for_simulation[1, 0] == lp_image_pixel_7
            assert plane.profile_image_plane_image_2d_for_simulation[1, 1] == lp_image_pixel_8
            assert plane.profile_image_plane_image_2d_for_simulation[1, 2] == lp_image_pixel_9
            assert plane.profile_image_plane_image_2d_for_simulation[1, 3] == lp_image_pixel_10

        def test__plane_has_no_galaxies__image_is_zeros_size_of_unlensed_regular_grid(self, grid_stack_5x5):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None,
                                            redshift=None)

            assert plane.profile_image_plane_image_2d.shape == (5, 5)
            assert (plane.profile_image_plane_image_2d[1, 1] == 0.0).all()
            assert (plane.profile_image_plane_image_2d[1, 2] == 0.0).all()

    class TestBlurringImage:

        def test__image_from_plane__same_as_its_light_profile_image(self, grid_stack_5x5, gal_x1_lp):
            light_profile = gal_x1_lp.light_profiles[0]

            lp_blurring_image = light_profile.intensities_from_grid(grid_stack_5x5.blurring)

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_lp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_blurring_image_1d == lp_blurring_image).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_blurring_image = lp0.intensities_from_grid(grid_stack_5x5.blurring)
            lp1_blurring_image = lp1.intensities_from_grid(grid_stack_5x5.blurring)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_blurring_image_1d == lp0_blurring_image + lp1_blurring_image).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grid_stack_5x5, gal_x1_lp):
            galaxy_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.blurring, galaxies=[gal_x1_lp])

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_lp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_blurring_image_1d == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.blurring, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.blurring, galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_blurring_image_1d == g0_image + g1_image).all()

        def test__image_plane_blurring_image_1d_of_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.blurring, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack_5x5.blurring, galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.profile_image_plane_blurring_image_1d_of_galaxies[0] == g0_image).all()
            assert (plane.profile_image_plane_blurring_image_1d_of_galaxies[1] == g1_image).all()

    class TestConvergence:

        def test__convergence_from_plane__same_as_its_mass_profile(self, grid_stack_5x5, gal_x1_mp):
            mass_profile = gal_x1_mp.mass_profiles[0]

            mp_sub_convergence = mass_profile.convergence_from_grid(grid=grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp_convergence_pixel_0 = (mp_sub_convergence[0] + mp_sub_convergence[1] +
                                      mp_sub_convergence[2] + mp_sub_convergence[3]) / 4
            mp_convergence_pixel_1 = (mp_sub_convergence[4] + mp_sub_convergence[5] +
                                      mp_sub_convergence[6] + mp_sub_convergence[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.convergence.shape == (5, 5)
            assert (plane.convergence[1, 1] == mp_convergence_pixel_0).all()
            assert (plane.convergence[1, 2] == mp_convergence_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0, centre=(1.0, 0.0)))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0, centre=(1.0, 1.0)))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_convergence = mp0.convergence_from_grid(grid=grid_stack_5x5.sub.unlensed_sub_grid)
            mp1_sub_convergence = mp1.convergence_from_grid(grid=grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp0_convergence_pixel_0 = (mp0_sub_convergence[0] + mp0_sub_convergence[1] +
                                       mp0_sub_convergence[2] + mp0_sub_convergence[3]) / 4
            mp0_convergence_pixel_1 = (mp0_sub_convergence[4] + mp0_sub_convergence[5] +
                                       mp0_sub_convergence[6] + mp0_sub_convergence[7]) / 4
            mp1_convergence_pixel_0 = (mp1_sub_convergence[0] + mp1_sub_convergence[1] +
                                       mp1_sub_convergence[2] + mp1_sub_convergence[3]) / 4
            mp1_convergence_pixel_1 = (mp1_sub_convergence[4] + mp1_sub_convergence[5] +
                                       mp1_sub_convergence[6] + mp1_sub_convergence[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.convergence[1, 1] == pytest.approx(mp0_convergence_pixel_0 +
                                                            mp1_convergence_pixel_0, 1.0e-4)
            assert plane.convergence[1, 2] == pytest.approx(mp0_convergence_pixel_1 +
                                                            mp1_convergence_pixel_1, 1.0e-4)

        def test__convergence__same_as_its_galaxy(self, grid_stack_5x5, gal_x1_mp):
            galaxy_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack_5x5.sub.unlensed_sub_grid,
                                                                               galaxies=[gal_x1_mp])

            galaxy_convergence = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(galaxy_convergence)

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.convergence == galaxy_convergence).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack_5x5):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_convergence = galaxy_util.convergence_of_galaxies_from_grid(
                grid=grid_stack_5x5.sub.unlensed_sub_grid, galaxies=[g0])

            g1_convergence = galaxy_util.convergence_of_galaxies_from_grid(
                grid=grid_stack_5x5.sub.unlensed_sub_grid, galaxies=[g1])

            g0_convergence = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(g0_convergence)
            g1_convergence = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(g1_convergence)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.convergence == pytest.approx(g0_convergence + g1_convergence, 1.0e-4)

        def test__plane_has_no_galaxies__convergence_is_zeros_size_of_unlensed_regular_grid(self, grid_stack_5x5):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.convergence.shape == (5, 5)
            assert (plane.convergence[1, 1] == 0.0).all()
            assert (plane.convergence[1, 2] == 0.0).all()

    class TestPotential:

        def test__potential_from_plane__same_as_its_mass_profile(self, grid_stack_5x5, gal_x1_mp):
            mass_profile = gal_x1_mp.mass_profiles[0]

            mp_sub_potential = mass_profile.potential_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp_potential_pixel_0 = (mp_sub_potential[0] + mp_sub_potential[1] + mp_sub_potential[2] + mp_sub_potential[
                3]) / 4
            mp_potential_pixel_1 = (mp_sub_potential[4] + mp_sub_potential[5] + mp_sub_potential[6] + mp_sub_potential[
                7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.potential.shape == (5, 5)
            assert (plane.potential[1, 1] == mp_potential_pixel_0).all()
            assert (plane.potential[1, 2] == mp_potential_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub.unlensed_sub_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_potential = mp0.potential_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)
            mp1_sub_potential = mp1.potential_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp0_potential_pixel_0 = (mp0_sub_potential[0] + mp0_sub_potential[1] +
                                     mp0_sub_potential[2] + mp0_sub_potential[3]) / 4
            mp0_potential_pixel_1 = (mp0_sub_potential[4] + mp0_sub_potential[5] +
                                     mp0_sub_potential[6] + mp0_sub_potential[7]) / 4
            mp1_potential_pixel_0 = (mp1_sub_potential[0] + mp1_sub_potential[1] +
                                     mp1_sub_potential[2] + mp1_sub_potential[3]) / 4
            mp1_potential_pixel_1 = (mp1_sub_potential[4] + mp1_sub_potential[5] +
                                     mp1_sub_potential[6] + mp1_sub_potential[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.potential[1, 1] == pytest.approx(mp0_potential_pixel_0 +
                                                          mp1_potential_pixel_0, 1.0e-4)
            assert plane.potential[1, 2] == pytest.approx(mp0_potential_pixel_1 +
                                                          mp1_potential_pixel_1, 1.0e-4)

        def test__potential__same_as_its_galaxy(self, grid_stack_5x5, gal_x1_mp):
            galaxy_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack_5x5.sub.unlensed_sub_grid,
                                                                           galaxies=[gal_x1_mp])

            galaxy_potential = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(galaxy_potential)

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.potential == galaxy_potential).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub.unlensed_sub_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack_5x5.sub.unlensed_sub_grid,
                                                                       galaxies=[g0])
            g1_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack_5x5.sub.unlensed_sub_grid,
                                                                       galaxies=[g1])

            g0_potential = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(g0_potential)
            g1_potential = grid_stack_5x5.regular.scaled_array_2d_from_array_1d(g1_potential)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.potential == pytest.approx(g0_potential + g1_potential, 1.0e-4)

        def test__plane_has_no_galaxies__potential_are_zeros_size_of_unlensed_regular_grid(self, grid_stack_5x5):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.potential.shape == (5, 5)
            assert (plane.potential[1, 1] == 0.0).all()
            assert (plane.potential[1, 2] == 0.0).all()

    class TestDeflections:

        def test__deflections_from_plane__same_as_its_mass_profile(self, grid_stack_5x5, gal_x1_mp):
            mp = gal_x1_mp.mass_profiles[0]

            mp_sub_image = mp.deflections_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp_image_pixel_0x = (mp_sub_image[0, 0] + mp_sub_image[1, 0] + mp_sub_image[2, 0] + mp_sub_image[3, 0]) / 4
            mp_image_pixel_1x = (mp_sub_image[4, 0] + mp_sub_image[5, 0] + mp_sub_image[6, 0] + mp_sub_image[7, 0]) / 4
            mp_image_pixel_0y = (mp_sub_image[0, 1] + mp_sub_image[1, 1] + mp_sub_image[2, 1] + mp_sub_image[3, 1]) / 4
            mp_image_pixel_1y = (mp_sub_image[4, 1] + mp_sub_image[5, 1] + mp_sub_image[6, 1] + mp_sub_image[7, 1]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.deflections_1d[0, 0] == mp_image_pixel_0x).all()
            assert (plane.deflections_1d[0, 1] == mp_image_pixel_0y).all()
            assert (plane.deflections_1d[1, 0] == mp_image_pixel_1x).all()
            assert (plane.deflections_1d[1, 1] == mp_image_pixel_1y).all()

            assert plane.deflections_y.shape == (5, 5)
            assert plane.deflections_x.shape == (5, 5)
            assert (plane.deflections_y ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub.unlensed_sub_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_image = mp0.deflections_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)
            mp1_sub_image = mp1.deflections_from_grid(grid_stack_5x5.sub.unlensed_sub_grid)

            # Perform sub gridding average manually
            mp0_image_pixel_0x = (mp0_sub_image[0, 0] + mp0_sub_image[1, 0] +
                                  mp0_sub_image[2, 0] + mp0_sub_image[3, 0]) / 4
            mp0_image_pixel_1x = (mp0_sub_image[4, 0] + mp0_sub_image[5, 0] +
                                  mp0_sub_image[6, 0] + mp0_sub_image[7, 0]) / 4
            mp0_image_pixel_0y = (mp0_sub_image[0, 1] + mp0_sub_image[1, 1] +
                                  mp0_sub_image[2, 1] + mp0_sub_image[3, 1]) / 4
            mp0_image_pixel_1y = (mp0_sub_image[4, 1] + mp0_sub_image[5, 1] +
                                  mp0_sub_image[6, 1] + mp0_sub_image[7, 1]) / 4

            mp1_image_pixel_0x = (mp1_sub_image[0, 0] + mp1_sub_image[1, 0] +
                                  mp1_sub_image[2, 0] + mp1_sub_image[3, 0]) / 4
            mp1_image_pixel_1x = (mp1_sub_image[4, 0] + mp1_sub_image[5, 0] +
                                  mp1_sub_image[6, 0] + mp1_sub_image[7, 0]) / 4
            mp1_image_pixel_0y = (mp1_sub_image[0, 1] + mp1_sub_image[1, 1] +
                                  mp1_sub_image[2, 1] + mp1_sub_image[3, 1]) / 4
            mp1_image_pixel_1y = (mp1_sub_image[4, 1] + mp1_sub_image[5, 1] +
                                  mp1_sub_image[6, 1] + mp1_sub_image[7, 1]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.deflections_1d[0, 0] == pytest.approx(mp0_image_pixel_0x + mp1_image_pixel_0x, 1.0e-4)
            assert plane.deflections_1d[1, 0] == pytest.approx(mp0_image_pixel_1x + mp1_image_pixel_1x, 1.0e-4)
            assert plane.deflections_1d[0, 1] == pytest.approx(mp0_image_pixel_0y + mp1_image_pixel_0y, 1.0e-4)
            assert plane.deflections_1d[1, 1] == pytest.approx(mp0_image_pixel_1y + mp1_image_pixel_1y, 1.0e-4)
            assert (plane.deflections_y ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__deflections__same_as_its_galaxy(self, grid_stack_5x5, gal_x1_mp):
            galaxy_deflections = galaxy_util.deflections_of_galaxies_from_grid(
                grid=grid_stack_5x5.sub.unlensed_sub_grid,
                galaxies=[gal_x1_mp])

            plane = pl.AbstractGriddedPlane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.deflections_1d == galaxy_deflections).all()
            assert (plane.deflections_y ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack_5x5):
            # Overwrite one value so intensity in each pixel is different
            grid_stack_5x5.sub.unlensed_sub_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack_5x5.sub.unlensed_sub_grid,
                                                                           galaxies=[g0])
            g1_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack_5x5.sub.unlensed_sub_grid,
                                                                           galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.deflections_1d == pytest.approx(g0_deflections + g1_deflections, 1.0e-4)
            assert (plane.deflections_y ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__plane_has_no_galaxies__deflections_are_zeros_size_of_unlensed_regular_grid(self, grid_stack_5x5):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert (plane.deflections_1d[0, 0] == 0.0).all()
            assert (plane.deflections_1d[0, 1] == 0.0).all()
            assert (plane.deflections_1d[1, 0] == 0.0).all()
            assert (plane.deflections_1d[1, 1] == 0.0).all()

            assert plane.deflections_y.shape == (5, 5)
            assert plane.deflections_x.shape == (5, 5)
            assert (plane.deflections_y ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack_5x5.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

    class TestMapper:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, grid_stack_5x5):
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix], grid_stack=grid_stack_5x5,
                                            compute_deflections=False,
                                            border=[mock_grids.MockBorders()], redshift=None)

            assert plane.mapper is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, grid_stack_5x5):

            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                  regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_pix], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=[mock_grids.MockBorders()], redshift=None)

            assert plane.mapper == 1

            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                  regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix, galaxy_pix], grid_stack=grid_stack_5x5,
                                            compute_deflections=False, border=[mock_grids.MockBorders()], redshift=None)

            assert plane.mapper == 1

        def test__plane_has_no_border__still_returns_mapper(self, grid_stack_5x5):
            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                  regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix, galaxy_pix], grid_stack=grid_stack_5x5,
                                            compute_deflections=False, border=None, redshift=None)

            assert plane.mapper == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, grid_stack_5x5):
            galaxy_pix_0 = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=1),
                                    regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))
            galaxy_pix_1 = g.Galaxy(redshift=0.5, pixelization=mock_inv.MockPixelization(value=2),
                                    regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_pix_0, galaxy_pix_1], grid_stack=grid_stack_5x5,
                                            compute_deflections=False, border=[mock_grids.MockBorders()], redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.mapper)

    class TestProperties:

        def test__padded_grid_in__tracer_has_padded_grid_property(self, grid_stack_5x5, padded_grid_stack_5x5,
                                                                  gal_x1_lp):
            plane = pl.AbstractGriddedPlane(grid_stack=grid_stack_5x5, galaxies=[gal_x1_lp], compute_deflections=False,
                                            redshift=None, border=None)
            assert plane.has_padded_grid_stack is False

            plane = pl.AbstractGriddedPlane(grid_stack=padded_grid_stack_5x5, galaxies=[gal_x1_lp],
                                            compute_deflections=False,
                                            redshift=None, border=None)
            assert plane.has_padded_grid_stack is True

            mask = msk.Mask(np.array([[True, False]]), pixel_scale=3.0)

            padded_grid_stack_5x5 = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask, 2,
                                                                                                            (3, 3))

            plane = pl.AbstractGriddedPlane(grid_stack=padded_grid_stack_5x5, galaxies=[gal_x1_lp],
                                            compute_deflections=False,
                                            redshift=None, border=None)
            assert plane.has_padded_grid_stack is True

    class TestPlaneImage:

        def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self, grid_stack_5x5):
            grid_stack_5x5.regular[1] = np.array([2.0, 2.0])

            galaxy = g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            plane_image_from_func = lens_util.plane_image_of_galaxies_from_grid(shape=(5, 5),
                                                                                grid=grid_stack_5x5.regular,
                                                                                galaxies=[galaxy])

            assert (plane_image_from_func == plane.plane_image).all()

        def test__ensure_index_of_plane_image_has_negative_arcseconds_at_start(self, grid_stack_5x5):
            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The origin -1.6, -1.6 of the model_galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane datas_.

            mask = msk.Mask(array=np.full((5, 5), False), pixel_scale=1.0)

            grid_stack_5x5.regular = grids.RegularGrid(np.array([[-2.0, -2.0], [2.0, 2.0]]), mask=mask)

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.plane_image.shape == (5, 5)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (0, 0)

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (0, 4)

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(-1.6, -1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (4, 0)

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack_5x5, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (4, 4)


class TestAbstractDataPlane(object):

    class TestBlurredImagePlaneImage:

        def test__blurred_images_1d_of_galaxies(self, grid_stack_5x5, convolver_image_5x5):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image_1d = galaxy_util.intensities_of_galaxies_from_grid(galaxies=[g0], grid=grid_stack_5x5.sub)
            g0_blurring_image_1d = galaxy_util.intensities_of_galaxies_from_grid(galaxies=[g0],
                                                                                 grid=grid_stack_5x5.blurring)

            g1_image_1d = galaxy_util.intensities_of_galaxies_from_grid(galaxies=[g1], grid=grid_stack_5x5.sub)
            g1_blurring_image_1d = galaxy_util.intensities_of_galaxies_from_grid(galaxies=[g1],
                                                                                 grid=grid_stack_5x5.blurring)

            blurred_g0_image = convolver_image_5x5.convolve_image(image_array=g0_image_1d,
                                                                  blurring_array=g0_blurring_image_1d)

            blurred_g1_image = convolver_image_5x5.convolve_image(image_array=g1_image_1d,
                                                                  blurring_array=g1_blurring_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[g0, g1], grid_stack=grid_stack_5x5,
                                         compute_deflections=False,
                                         border=None)

            blurred_images_1d_of_galaxies = \
                plane.blurred_profile_image_plane_images_1d_of_galaxies_from_convolver_image(
                    convolver_image=convolver_image_5x5)

            assert (blurred_images_1d_of_galaxies[0] == blurred_g0_image).all()
            assert (blurred_images_1d_of_galaxies[1] == blurred_g1_image).all()

            blurred_image_1d = \
                plane.blurred_profile_image_plane_image_1d_from_convolver_image(convolver_image=convolver_image_5x5)

            assert blurred_image_1d == pytest.approx(blurred_g0_image + blurred_g1_image, 1.0e-4)

    class TestUnmaskedBlurrerImagePlaneImage:

        def test__unmasked_blurred_images_1d_of_galaxies(self):
            psf = ccd.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                           [0.0, 1.0, 2.0],
                                           [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                            [True, False, True],
                                            [True, True, True]]), pixel_scale=1.0)

            padded_grid_stack_5x5 = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
                mask=mask, sub_grid_size=1, psf_shape=(3, 3))

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.2))

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[g0, g1], grid_stack=padded_grid_stack_5x5,
                                         compute_deflections=False, border=None)

            manual_blurred_image_0 = plane.profile_image_plane_image_1d_of_galaxies[0]
            manual_blurred_image_0 = padded_grid_stack_5x5.regular.padded_array_2d_from_padded_array_1d(
                padded_array_1d=manual_blurred_image_0)
            manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

            manual_blurred_image_1 = plane.profile_image_plane_image_1d_of_galaxies[1]
            manual_blurred_image_1 = padded_grid_stack_5x5.regular.padded_array_2d_from_padded_array_1d(
                padded_array_1d=manual_blurred_image_1)
            manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

            unmasked_blurred_image_plane_images = plane.unmasked_blurred_profile_image_plane_images_of_galaxies_from_psf(
                psf=psf)

            assert (unmasked_blurred_image_plane_images[0] == manual_blurred_image_0[1:4, 1:4]).all()
            assert (unmasked_blurred_image_plane_images[1] == manual_blurred_image_1[1:4, 1:4]).all()

            unmasked_blurred_image_plane_image = plane.unmasked_blurred_profile_image_plane_image_from_psf(psf=psf)

            assert unmasked_blurred_image_plane_image == \
                   pytest.approx(manual_blurred_image_0[1:4, 1:4] + manual_blurred_image_1[1:4, 1:4], 1.0e-4)

    class TestContributionMaps:

        def test__x2_hyper_galaxy__use_numerical_values_for_noise_scaling(self):

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)

            hyper_model_image_1d = np.array([0.5, 1.0, 1.5])

            hyper_galaxy_image_0 = np.array([0.5, 1.0, 1.5])
            hyper_galaxy_image_1 = np.array([0.5, 1.0, 1.5])

            galaxy_0 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_0)

            galaxy_1 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0, galaxy_1], grid_stack=None,
                                         compute_deflections=False, border=None)

            assert (plane.contribution_maps_1d_of_galaxies[0] == np.array([1.0, 1.0, 1.0])).all()
            assert (plane.contribution_maps_1d_of_galaxies[1] == np.array([5.0/9.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

        def test__contribution_maps_are_same_as_hyper_galaxy_calculation(self):

            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=10.0)

            contribution_map_1d_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            contribution_map_1d_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            galaxy_0 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            galaxy_1 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0], grid_stack=None, compute_deflections=False,
                                         border=None)

            assert (plane.contribution_maps_1d_of_galaxies[0] == contribution_map_1d_0).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1], grid_stack=None, compute_deflections=False,
                                         border=None)

            assert (plane.contribution_maps_1d_of_galaxies[0] == contribution_map_1d_1).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1, galaxy_0], grid_stack=None,
                                         compute_deflections=False, border=None)

            assert (plane.contribution_maps_1d_of_galaxies[0] == contribution_map_1d_1).all()
            assert (plane.contribution_maps_1d_of_galaxies[1] == contribution_map_1d_0).all()

        def test__contriution_maps_are_none_for_galaxy_without_hyper_galaxy(self):
            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy = g.HyperGalaxy(contribution_factor=5.0)

            contribution_map_1d = hyper_galaxy.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            galaxy = g.Galaxy(redshift=0.5,
                              hyper_galaxy=hyper_galaxy, hyper_model_image_1d=hyper_model_image_1d,
                              hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5,
                                         galaxies=[galaxy, g.Galaxy(redshift=0.5), g.Galaxy(redshift=0.5)],
                                         grid_stack=None, compute_deflections=False, border=None)

            assert (plane.contribution_maps_1d_of_galaxies[0] == contribution_map_1d).all()
            assert plane.contribution_maps_1d_of_galaxies[1] == None
            assert plane.contribution_maps_1d_of_galaxies[2] == None

    class TestHyperNoiseMap:

        def test__x2_hyper_galaxy__use_numerical_values_of_hyper_noise_map_scaling(self):

            noise_map_1d = np.array([1.0, 2.0, 3.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=0.0, noise_factor=1.0, noise_power=1.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=3.0, noise_factor=1.0, noise_power=2.0)

            hyper_model_image_1d = np.array([0.5, 1.0, 1.5])

            hyper_galaxy_image_1d_0 = np.array([0.0, 1.0, 1.5])
            hyper_galaxy_image_1d_1 = np.array([1.0, 1.0, 1.5])

            galaxy_0 = g.Galaxy(redshift=0.5,
                                hyper_galaxy=hyper_galaxy_0,
                                hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d_0)

            galaxy_1 = g.Galaxy(redshift=0.5,
                                hyper_galaxy=hyper_galaxy_1,
                                hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d_1)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0, galaxy_1], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)

            assert (hyper_noise_maps_1d[0] == np.array([0.0, 2.0, 3.0])).all()
            assert hyper_noise_maps_1d[1] == pytest.approx(np.array([0.73468, (2.0 * 0.75) ** 2.0, 3.0 ** 2.0]), 1.0e-4)

        def test__hyper_noise_maps_1d_are_same_as_hyper_galaxy_calculation(self):
            noise_map_1d = np.array([5.0, 3.0, 1.0])

            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=10.0)

            contribution_map_1d_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            contribution_map_1d_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            hyper_noise_map_1d_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_0)

            hyper_noise_map_1d_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_1)

            galaxy_0 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            galaxy_1 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0], grid_stack=None, compute_deflections=False,
                                         border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_0).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1], grid_stack=None, compute_deflections=False,
                                         border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_1).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1, galaxy_0], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_1).all()
            assert (hyper_noise_maps_1d[1] == hyper_noise_map_1d_0).all()

        def test__hyper_noise_maps_1d_are_none_for_galaxy_without_hyper_galaxy(self):
            noise_map_1d = np.array([5.0, 3.0, 1.0])

            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=10.0)

            contribution_map_1d_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            contribution_map_1d_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            hyper_noise_map_1d_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_0)

            hyper_noise_map_1d_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_1)

            galaxy_0 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            galaxy_1 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0, g.Galaxy(redshift=0.5)], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_0).all()
            assert hyper_noise_maps_1d[1] == None

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[g.Galaxy(redshift=0.5), galaxy_1], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert hyper_noise_maps_1d[0] == None
            assert (hyper_noise_maps_1d[1] == hyper_noise_map_1d_1).all()

            plane = pl.AbstractDataPlane(redshift=0.5,
                                         galaxies=[g.Galaxy(redshift=0.5), galaxy_1, galaxy_0, g.Galaxy(redshift=0.5)],
                                         grid_stack=None, compute_deflections=False, border=None)

            hyper_noise_maps_1d = plane.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert hyper_noise_maps_1d[0] == None
            assert (hyper_noise_maps_1d[1] == hyper_noise_map_1d_1).all()
            assert (hyper_noise_maps_1d[2] == hyper_noise_map_1d_0).all()
            assert hyper_noise_maps_1d[3] == None

        def test__hyper_noise_map_from_noise_map__is_sum_of_galaxy_hyper_noise_maps_1d__filters_nones(self):
            noise_map_1d = np.array([5.0, 3.0, 1.0])

            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=10.0)

            contribution_map_1d_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            contribution_map_1d_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image_1d, hyper_galaxy_image=hyper_galaxy_image_1d)

            hyper_noise_map_1d_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_0)

            hyper_noise_map_1d_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map_1d, contribution_map=contribution_map_1d_1)

            galaxy_0 = g.Galaxy(redshift=0.5,
                                hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            galaxy_1 = g.Galaxy(redshift=0.5,
                                hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d)

            plane = pl.AbstractDataPlane(redshift=0.5,
                                         galaxies=[galaxy_0], grid_stack=None, compute_deflections=False, border=None)

            hyper_noise_map_1d = plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_map_1d == hyper_noise_map_1d_0).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1], grid_stack=None, compute_deflections=False,
                                         border=None)

            hyper_noise_map_1d = plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_map_1d == hyper_noise_map_1d_1).all()

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1, galaxy_0], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_map_1d = plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_map_1d == hyper_noise_map_1d_0 + hyper_noise_map_1d_1).all()

            plane = pl.AbstractDataPlane(redshift=0.5,
                                         galaxies=[g.Galaxy(redshift=0.5), galaxy_1, galaxy_0, g.Galaxy(redshift=0.5)],
                                         grid_stack=None, compute_deflections=False, border=None)

            hyper_noise_map_1d = plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert (hyper_noise_map_1d == hyper_noise_map_1d_0 + hyper_noise_map_1d_1).all()

        def test__plane_has_no_hyper_galaxies__hyper_noise_map_function_returns_none(self):

            noise_map_1d = np.array([5.0, 3.0, 1.0])

            plane = pl.AbstractDataPlane(redshift=0.5, galaxies=[g.Galaxy(redshift=0.5)], grid_stack=None,
                                         compute_deflections=False, border=None)
            hyper_noise_map_1d = plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            assert hyper_noise_map_1d == 0


class TestPlane(object):
    class TestGridLensing:

        def test__grid_stack_setup_for_regular_sub_and_blurring__no_deflections(self, grid_stack_5x5, gal_x1_mp):
            plane = pl.Plane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5,
                             compute_deflections=False, redshift=None, border=None)

            assert plane.grid_stack.regular[0:2] == pytest.approx(np.array([[1.0, -1.0], [1.0, 0.0]]), 1e-3)

            assert plane.grid_stack.sub[0:8] == \
                   pytest.approx(np.array([[1.25, -1.25], [1.25, -0.75], [0.75, -1.25], [0.75, -0.75],
                                           [1.25, -0.25], [1.25, 0.25], [0.75, -0.25], [0.75, 0.25]]), 1e-3)
            assert plane.grid_stack.blurring == pytest.approx(
                np.array([[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [1.0, -2.0], [1.0, 2.0],
                          [0.0, -2.0], [0.0, 2.0], [-1.0, -2.0], [-1.0, 2.0], [-2.0, -2.0], [-2.0, -1.0], [-2.0, 0.0],
                          [-2.0, 1.0], [-2.0, 2.0]]), 1e-3)

            assert plane.deflections_stack is None

        def test__same_as_above_but_test_deflections(self, grid_stack_5x5, gal_x1_mp):
            plane = pl.Plane(galaxies=[gal_x1_mp], grid_stack=grid_stack_5x5,
                             compute_deflections=True, redshift=None, border=None)

            sub_galaxy_deflections = gal_x1_mp.deflections_from_grid(grid=grid_stack_5x5.sub)
            blurring_galaxy_deflections = gal_x1_mp.deflections_from_grid(grid=grid_stack_5x5.blurring)

            assert plane.deflections_stack.regular[0:2] == pytest.approx(np.array([[0.707, -0.707], [1.0, 0.0]]), 1e-3)
            assert (plane.deflections_stack.sub == sub_galaxy_deflections).all()
            assert (plane.deflections_stack.blurring == blurring_galaxy_deflections).all()

        def test__same_as_above__x2_galaxy_in_plane__or_galaxy_x2_sis__deflections_double(self,
                                                                                          grid_stack_5x5, gal_x1_mp,
                                                                                          gal_x2_mp):
            plane = pl.Plane(galaxies=[gal_x2_mp], grid_stack=grid_stack_5x5,
                             compute_deflections=True, redshift=None, border=None)

            sub_galaxy_deflections = gal_x2_mp.deflections_from_grid(grid_stack_5x5.sub)
            blurring_galaxy_deflections = gal_x2_mp.deflections_from_grid(grid_stack_5x5.blurring)

            assert plane.deflections_stack.regular[0:2] == \
                   pytest.approx(np.array([[3.0 * 0.707, -3.0 * 0.707], [3.0, 0.0]]),
                                 1e-3)
            assert (plane.deflections_stack.sub == sub_galaxy_deflections).all()
            assert (plane.deflections_stack.blurring == blurring_galaxy_deflections).all()

            plane = pl.Plane(galaxies=[gal_x1_mp, gal_x1_mp], grid_stack=grid_stack_5x5,
                             compute_deflections=True, redshift=None, border=None)

            sub_galaxy_deflections = gal_x1_mp.deflections_from_grid(grid_stack_5x5.sub)
            blurring_galaxy_deflections = gal_x1_mp.deflections_from_grid(grid_stack_5x5.blurring)

            assert plane.deflections_stack.regular[0:2] == pytest.approx(
                np.array([[2.0 * 0.707, -2.0 * 0.707], [2.0, 0.0]]),
                1e-3)
            assert (plane.deflections_stack.sub == 2.0 * sub_galaxy_deflections).all()
            assert (plane.deflections_stack.blurring == 2.0 * blurring_galaxy_deflections).all()

        def test__plane_has_no_galaxies__deflections_all_zeros_shape_of_grid_stack_5x5(self, grid_stack_5x5):
            plane = pl.Plane(galaxies=[], grid_stack=grid_stack_5x5, border=None,
                             compute_deflections=True, redshift=1.0)

            assert (plane.deflections_stack.regular[0:2] == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (plane.deflections_stack.sub[0:8] == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0]])).all()
            assert (plane.deflections_stack.blurring == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                  [0.0, 0.0]])).all()

    class TestGalaxies:

        def test__no_galaxies__raises_exception_if_no_plane_redshift_input(self):
            plane = pl.Plane(galaxies=[], grid_stack=None, compute_deflections=False, redshift=0.5)
            assert plane.redshift == 0.5

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[], grid_stack=None, compute_deflections=False)

        def test__galaxy_redshifts_gives_list_of_redshifts(self):
            g0 = g.Galaxy(redshift=1.0)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=1.0)

            plane = pl.Plane(galaxies=[g0, g1, g2], grid_stack=None, compute_deflections=False)

            assert plane.redshift == 1.0
            assert plane.galaxy_redshifts == [1.0, 1.0, 1.0]

        def test__galaxies_have_different_redshifts__exception_is_raised_if_redshift_not_input(self):
            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)

            g0 = g.Galaxy(redshift=0.4)
            g1 = g.Galaxy(redshift=0.5)
            g2 = g.Galaxy(redshift=0.6)

            plane = pl.Plane(galaxies=[g0, g1, g2], grid_stack=None, compute_deflections=False, redshift=0.5)

            assert plane.redshift == 0.5


class TestPlaneImage:

    def test__compute_xticks_from_regular_grid_correctly(self):
        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(5.0, 1.0), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(5.0, 0.5), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((1, 6)), pixel_scales=(5.0, 1.0), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2)

    def test__compute_yticks_from_regular_grid_correctly(self):
        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(1.0, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(0.5, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((6, 1)), pixel_scales=(1.0, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2)
