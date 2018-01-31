import galaxy
from profile import mass_profile, light_profile
import pytest


@pytest.fixture(name='sersic')
def circular_sersic():
    return light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                            sersic_index=4.0)


class TestProfiles(object):
    def test_intensity_at_coordinates(self, sersic):
        g = galaxy.Galaxy(1., sersic)
        intensity = g.intensity_at_coordinates((0., 1.0))
        assert intensity == pytest.approx(0.351797, 1e-3)

    def test_surface_density_at_coordinates(self, sersic):
        power_law = mass_profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                               einstein_radius=1.0, slope=2.3)

        g = galaxy.Galaxy(1., sersic, power_law)

        surface_density = g.surface_density_at_coordinates(coordinates=(1.0, 0.0))

        assert surface_density == pytest.approx(0.466666, 1e-3)


class TestGalaxyCollection(object):
    def test_trivial_ordering(self):
        galaxy_collection = galaxy.GalaxyCollection()
        g0 = galaxy.Galaxy(0)
        g1 = galaxy.Galaxy(1)
        g2 = galaxy.Galaxy(2)
        galaxy_collection.append(g0)
        galaxy_collection.append(g1)
        galaxy_collection.append(g2)

        assert galaxy_collection == [g0, g1, g2]

    def test_reverse_ordering(self):
        galaxy_collection = galaxy.GalaxyCollection()
        g0 = galaxy.Galaxy(0)
        g1 = galaxy.Galaxy(1)
        g2 = galaxy.Galaxy(2)
        galaxy_collection.append(g2)
        galaxy_collection.append(g1)
        galaxy_collection.append(g0)

        assert galaxy_collection == [g0, g1, g2]

    def test_out_of_order(self):
        galaxy_collection = galaxy.GalaxyCollection()
        g0 = galaxy.Galaxy(0)
        g1 = galaxy.Galaxy(1)
        g2 = galaxy.Galaxy(2)
        galaxy_collection.append(g0)
        galaxy_collection.append(g2)
        galaxy_collection.append(g1)

        assert galaxy_collection == [g0, g1, g2]
