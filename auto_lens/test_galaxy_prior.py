from auto_lens import galaxy_prior as gp
from auto_lens.profiles import mass_profiles, light_profiles


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls


class TestGalaxyPrior:
    def test_attach_to_model_mapper(self):
        galaxy_prior = gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                                      mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal])

        mapper = MockModelMapper()

        galaxy_prior.attach_to_model_mapper(mapper)

        assert len(mapper.classes) == 2
