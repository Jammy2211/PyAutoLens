from auto_lens.analysis import galaxy_prior as gp
from auto_lens.analysis import galaxy as g
from auto_lens.profiles import mass_profiles, light_profiles
from auto_lens.analysis import model_mapper as mm
import pytest
from auto_lens import exc
import os


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


@pytest.fixture(name='test_config')
def make_test_config():
    return mm.Config(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/config"))


@pytest.fixture(name="mapper")
def make_mapper():
    return mm.ModelMapper()


@pytest.fixture(name="galaxy_prior_2")
def make_galaxy_prior_2(mapper, test_config):
    galaxy_prior_2 = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                    mass_profile=mass_profiles.EllipticalCoredIsothermal, config=test_config)
    mapper.galaxy_2 = galaxy_prior_2
    return galaxy_prior_2


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior(mapper, test_config):
    galaxy_prior_1 = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                    mass_profile=mass_profiles.EllipticalCoredIsothermal, config=test_config)
    mapper.galaxy_1 = galaxy_prior_1
    return galaxy_prior_1


class TestGalaxyPrior:
    def test_init_to_model_mapper(self, mapper, test_config):
        mapper.galaxy_1 = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                         mass_profile=mass_profiles.EllipticalCoredIsothermal, config=test_config)
        assert len(mapper.priors_ordered_by_id) == 13

    def test_recover_classes(self, galaxy_prior):
        instance = MockModelInstance()

        light_profile_name = galaxy_prior.light_profile_names[0]
        mass_profile_name = galaxy_prior.mass_profile_names[0]
        redshift_name = galaxy_prior.redshift_name

        setattr(instance, light_profile_name, light_profiles.EllipticalDevVaucouleurs())
        setattr(instance, mass_profile_name, mass_profiles.EllipticalCoredIsothermal())
        setattr(instance, redshift_name, g.Redshift(1))

        galaxy = galaxy_prior.galaxy_for_model_instance(instance)

        assert len(galaxy.light_profiles) == 1
        assert len(galaxy.mass_profiles) == 1
        assert galaxy.redshift == 1

    def test_exceptions(self, galaxy_prior):
        instance = MockModelInstance()
        with pytest.raises(exc.PriorException):
            galaxy_prior.galaxy_for_model_instance(instance)

    def test_multiple_galaxies(self, mapper, test_config):
        mapper.galaxy_1 = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                         mass_profile=mass_profiles.EllipticalCoredIsothermal, config=test_config)
        mapper.galaxy_2 = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                         mass_profile=mass_profiles.EllipticalCoredIsothermal, config=test_config)
        assert len(mapper.prior_models) == 2

    def test_align_centres(self, galaxy_prior, mapper):
        prior_models = galaxy_prior.prior_models

        assert prior_models[0].centre != prior_models[1].centre

        prior_models = gp.GalaxyPrior("galaxy", mapper, light_profile=light_profiles.EllipticalDevVaucouleurs,
                                      mass_profile=mass_profiles.EllipticalCoredIsothermal,
                                      align_centres=True).prior_models

        assert prior_models[0].centre == prior_models[1].centre

    def test_align_phis(self, galaxy_prior, test_config):
        prior_models = galaxy_prior.prior_models

        assert prior_models[0].phi != prior_models[1].phi

        prior_models = gp.GalaxyPrior(light_profile=light_profiles.EllipticalDevVaucouleurs,
                                      mass_profile=mass_profiles.EllipticalCoredIsothermal,
                                      align_orientations=True, config=test_config).prior_models
        assert prior_models[0].phi == prior_models[1].phi


class TestNamedProfiles:
    def test_constructor(self, test_config):
        galaxy_prior = gp.GalaxyPrior(light_profile=light_profiles.EllipticalSersic,
                                      mass_profile=mass_profiles.EllipticalGeneralizedNFW, config=test_config)

        assert len(galaxy_prior.light_profile_names) == 1
        assert len(galaxy_prior.light_profile_classes) == 1

        assert len(galaxy_prior.mass_profile_names) == 1
        assert len(galaxy_prior.mass_profile_classes) == 1

    def test_get_prior_model(self):
        galaxy_prior = gp.GalaxyPrior(light_profile=light_profiles.EllipticalSersic,
                                      mass_profile=mass_profiles.EllipticalSersicMass)

        assert isinstance(galaxy_prior.light_profile, mm.PriorModel)
        assert isinstance(galaxy_prior.mass_profile, mm.PriorModel)

    def test_set_prior_model(self):
        mapper = mm.ModelMapper()
        galaxy_prior = gp.GalaxyPrior(light_profile=light_profiles.EllipticalSersic,
                                      mass_profile=mass_profiles.EllipticalSersicMass)

        mapper.galaxy = galaxy_prior

        assert 15 == len(mapper.priors_ordered_by_id)

        galaxy_prior.light_profile = mm.PriorModel(light_profiles.LightProfile)

        assert 8 == len(mapper.priors_ordered_by_id)
