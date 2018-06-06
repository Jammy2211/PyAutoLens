from auto_lens import galaxy_prior as gp
from auto_lens.profiles import mass_profiles, light_profiles
import pytest


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls


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
