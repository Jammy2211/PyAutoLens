from auto_lens import galaxy_prior as gp
from auto_lens.profiles import mass_profiles, light_profiles
import pytest


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls


class MockModelInstance:
    pass


@pytest.fixture(name="mapper")
def make_mapper():
    return MockModelMapper()


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                          mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal])


class TestGalaxyPrior:
    def test_attach_to_model_mapper(self, galaxy_prior, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)

        assert len(mapper.classes) == 2

    def test_recover_classes(self, galaxy_prior, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)

        instance = MockModelInstance()

        light_profile_name = galaxy_prior.light_profile_names[0]
        mass_profile_name = galaxy_prior.mass_profile_names[0]

        setattr(instance, light_profile_name, light_profiles.EllipticalDevVaucouleurs())
        setattr(instance, mass_profile_name, mass_profiles.EllipticalCoredIsothermal())

        galaxy = galaxy_prior.galaxy_for_model_instance(instance)

        assert len(galaxy.light_profiles) == 1
        assert len(galaxy.mass_profiles) == 1
