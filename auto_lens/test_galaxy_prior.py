from auto_lens import galaxy_prior as gp
from auto_lens.profiles import mass_profiles, light_profiles
import pytest


class MockPriorModel:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.centre = "centre for {}".format(name)
        self.phi = "phi for {}".format(name)


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls
        return MockPriorModel(name, cls)


class MockModelInstance:
    pass


@pytest.fixture(name="mapper")
def make_mapper():
    return MockModelMapper()


@pytest.fixture(name="galaxy_prior_2")
def make_galaxy_prior_2():
    return gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                          mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal])


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                          mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal])


class TestGalaxyPrior:
    def test_attach_to_model_mapper(self, galaxy_prior, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)

        assert len(mapper.classes) == 3

    def test_recover_classes(self, galaxy_prior, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)

        instance = MockModelInstance()

        light_profile_name = galaxy_prior.light_profile_names[0]
        mass_profile_name = galaxy_prior.mass_profile_names[0]
        redshift_name = galaxy_prior.redshift_name

        setattr(instance, light_profile_name, light_profiles.EllipticalDevVaucouleurs())
        setattr(instance, mass_profile_name, mass_profiles.EllipticalCoredIsothermal())
        setattr(instance, redshift_name, gp.Value(1))

        galaxy = galaxy_prior.galaxy_for_model_instance(instance)

        assert len(galaxy.light_profiles) == 1
        assert len(galaxy.mass_profiles) == 1
        assert galaxy.redshift == 1

    def test_exceptions(self, galaxy_prior, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)
        instance = MockModelInstance()
        with pytest.raises(gp.PriorException):
            galaxy_prior.galaxy_for_model_instance(instance)

    def test_multiple_galaxies(self, galaxy_prior, galaxy_prior_2, mapper):
        galaxy_prior.attach_to_model_mapper(mapper)
        galaxy_prior_2.attach_to_model_mapper(mapper)

        assert len(mapper.classes) == 6

    def test_align_centres(self, galaxy_prior, mapper):
        prior_models = galaxy_prior.attach_to_model_mapper(mapper)

        assert prior_models[0].centre != prior_models[1].centre

        galaxy_prior = gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                                      mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal],
                                      align_centres=True)
        prior_models = galaxy_prior.attach_to_model_mapper(mapper)
        assert prior_models[0].centre == prior_models[1].centre

    def test_align_phis(self, galaxy_prior, mapper):
        prior_models = galaxy_prior.attach_to_model_mapper(mapper)

        assert prior_models[0].phi != prior_models[1].phi

        galaxy_prior = gp.GalaxyPrior(light_profile_classes=[light_profiles.EllipticalDevVaucouleurs],
                                      mass_profile_classes=[mass_profiles.EllipticalCoredIsothermal],
                                      align_orientations=True)
        prior_models = galaxy_prior.attach_to_model_mapper(mapper)
        assert prior_models[0].phi == prior_models[1].phi
