import os

import pytest

import autofit as af
from autolens.model.galaxy import galaxy as g, galaxy_model as gp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import (
    mass_profiles,
    light_profiles,
    light_and_mass_profiles,
)


class MockPriorModel:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.centre = "origin for {}".format(name)
        self.phi = "phi for {}".format(name)


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls
        return MockPriorModel(name, cls)


class MockModelInstance:
    pass


@pytest.fixture(name="mass_and_light")
def make_profile():
    return light_and_mass_profiles.EllipticalSersicRadialGradient()


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../../test_files/config/galaxy_model".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="galaxy_model_2")
def make_galaxy_model_2(mapper,):
    galaxy_model_2 = gp.GalaxyModel(
        redshift=g.Redshift,
        light_profile=light_profiles.EllipticalDevVaucouleurs,
        mass_profile=mass_profiles.EllipticalCoredIsothermal,
    )
    mapper.galaxy_2 = galaxy_model_2
    return galaxy_model_2


@pytest.fixture(name="galaxy_model")
def make_galaxy_model(mapper,):
    galaxy_model_1 = gp.GalaxyModel(
        redshift=g.Redshift,
        light_profile=light_profiles.EllipticalDevVaucouleurs,
        mass_profile=mass_profiles.EllipticalCoredIsothermal,
    )
    mapper.galaxy_1 = galaxy_model_1
    return galaxy_model_1


class TestMassAndLightProfiles(object):
    def test_make_galaxy_from_constant_profile(self, mass_and_light):
        prior = gp.GalaxyModel(redshift=0.5, profile=mass_and_light)

        galaxy = prior.instance_for_arguments({})

        assert galaxy.light_profiles[0] == mass_and_light
        assert galaxy.mass_profiles[0] == mass_and_light

    def test_make_galaxy_from_variable_profile(self):
        galaxy_model = gp.GalaxyModel(
            redshift=0.5, profile=light_and_mass_profiles.EllipticalSersic
        )

        arguments = {
            galaxy_model.profile.centre.centre_0: 1.0,
            galaxy_model.profile.centre.centre_1: 0.2,
            galaxy_model.profile.axis_ratio: 0.4,
            galaxy_model.profile.phi: 0.5,
            galaxy_model.profile.intensity.value: 0.6,
            galaxy_model.profile.effective_radius.value: 0.7,
            galaxy_model.profile.sersic_index: 0.8,
            galaxy_model.profile.mass_to_light_ratio.value: 3.0,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.light_profiles[0] == galaxy.mass_profiles[0]
        assert isinstance(
            galaxy.light_profiles[0], light_and_mass_profiles.EllipticalSersic
        )

        assert galaxy.mass_profiles[0].centre == (1.0, 0.2)
        assert galaxy.mass_profiles[0].axis_ratio == 0.4
        assert galaxy.mass_profiles[0].phi == 0.5
        assert galaxy.mass_profiles[0].intensity == 0.6
        assert galaxy.mass_profiles[0].effective_radius == 0.7
        assert galaxy.mass_profiles[0].sersic_index == 0.8
        assert galaxy.mass_profiles[0].mass_to_light_ratio == 3.0


class TestGalaxyModel:
    def test_init_to_model_mapper(self, mapper):
        mapper.galaxy_1 = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
        )
        print(mapper.galaxy_1.redshift)
        assert len(mapper.prior_tuples_ordered_by_id) == 13

    def test_multiple_galaxies(self, mapper):
        mapper.galaxy_1 = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
        )
        mapper.galaxy_2 = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
        )
        assert len(mapper.prior_model_tuples) == 2

    def test_align_centres(self, galaxy_model):
        prior_models = galaxy_model.prior_models

        assert prior_models[1].centre != prior_models[2].centre

        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
            align_centres=True,
        )

        prior_models = galaxy_model.prior_models

        assert prior_models[1].centre == prior_models[2].centre

    def test_align_axis_ratios(self, galaxy_model):
        prior_models = galaxy_model.prior_models

        assert prior_models[1].axis_ratio != prior_models[2].axis_ratio

        prior_models = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
            align_axis_ratios=True,
        ).prior_models
        assert prior_models[1].axis_ratio == prior_models[2].axis_ratio

    def test_align_phis(self, galaxy_model):
        prior_models = galaxy_model.prior_models

        assert prior_models[1].phi != prior_models[2].phi

        prior_models = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalDevVaucouleurs,
            mass_profile=mass_profiles.EllipticalCoredIsothermal,
            align_orientations=True,
        ).prior_models
        assert prior_models[1].phi == prior_models[2].phi


class TestNamedProfiles:
    def test_get_prior_model(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalSersic,
            mass_profile=mass_profiles.EllipticalSersic,
        )

        assert isinstance(galaxy_model.light_profile, af.PriorModel)
        assert isinstance(galaxy_model.mass_profile, af.PriorModel)

    def test_set_prior_model(self):
        mapper = af.ModelMapper()
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            light_profile=light_profiles.EllipticalSersic,
            mass_profile=mass_profiles.EllipticalSersic,
        )

        mapper.galaxy = galaxy_model

        assert 16 == len(mapper.prior_tuples_ordered_by_id)

        galaxy_model.light_profile = af.PriorModel(light_profiles.LightProfile)

        assert 9 == len(mapper.prior_tuples_ordered_by_id)


class TestResultForArguments:
    def test_simple_instance_for_arguments(self):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift)
        arguments = {galaxy_model.redshift.redshift: 0.5}
        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.redshift == 0.5

    def test_complicated_instance_for_arguments(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            align_centres=True,
            light_profile=light_profiles.EllipticalSersic,
            mass_profile=mass_profiles.SphericalIsothermal,
        )

        arguments = {
            galaxy_model.redshift.redshift: 0.5,
            galaxy_model.mass_profile.centre.centre_0: 1.0,
            galaxy_model.mass_profile.centre.centre_1: 0.2,
            galaxy_model.mass_profile.einstein_radius.value: 0.3,
            galaxy_model.light_profile.axis_ratio: 0.4,
            galaxy_model.light_profile.phi: 0.5,
            galaxy_model.light_profile.intensity.value: 0.6,
            galaxy_model.light_profile.effective_radius.value: 0.7,
            galaxy_model.light_profile.sersic_index: 2,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.light_profiles[0].centre[0] == 1.0
        assert galaxy.light_profiles[0].centre[1] == 0.2

    def test_gaussian_prior_model_for_arguments(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            align_centres=True,
            light_profile=light_profiles.EllipticalSersic,
            mass_profile=mass_profiles.SphericalIsothermal,
        )

        redshift_prior = af.GaussianPrior(1, 1)
        einstein_radius_prior = af.GaussianPrior(4, 1)
        intensity_prior = af.GaussianPrior(7, 1)

        arguments = {
            galaxy_model.redshift.redshift: redshift_prior,
            galaxy_model.mass_profile.centre.centre_0: af.GaussianPrior(2, 1),
            galaxy_model.mass_profile.centre.centre_1: af.GaussianPrior(3, 1),
            galaxy_model.mass_profile.einstein_radius.value: einstein_radius_prior,
            galaxy_model.light_profile.axis_ratio: af.GaussianPrior(5, 1),
            galaxy_model.light_profile.phi: af.GaussianPrior(6, 1),
            galaxy_model.light_profile.intensity.value: intensity_prior,
            galaxy_model.light_profile.effective_radius.value: af.GaussianPrior(8, 1),
            galaxy_model.light_profile.sersic_index: af.GaussianPrior(9, 1),
        }

        gaussian_galaxy_model_model = galaxy_model.gaussian_prior_model_for_arguments(
            arguments
        )

        assert gaussian_galaxy_model_model.redshift.redshift == redshift_prior
        assert (
            gaussian_galaxy_model_model.mass_profile.einstein_radius.value
            == einstein_radius_prior
        )
        assert (
            gaussian_galaxy_model_model.light_profile.intensity.value == intensity_prior
        )


class TestPixelization(object):
    def test_pixelization(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            pixelization=pix.Rectangular,
            regularization=reg.Constant,
        )

        arguments = {
            galaxy_model.redshift.redshift: 2.0,
            galaxy_model.pixelization.shape_0: 24.0,
            galaxy_model.pixelization.shape_1: 23.0,
            galaxy_model.regularization.coefficient: 0.5,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.pixelization.shape[0] == 24
        assert galaxy.pixelization.shape[1] == 23

    def test_fixed_pixelization(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            pixelization=pix.Rectangular(),
            regularization=reg.Constant(),
        )

        arguments = {galaxy_model.redshift.redshift: 2.0}

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.pixelization.shape[0] == 3
        assert galaxy.pixelization.shape[1] == 3

    def test__if_no_pixelization_raises_error(self):
        with pytest.raises(af.exc.PriorException):
            gp.GalaxyModel(redshift=g.Redshift, pixelization=pix.Voronoi)


class TestRegularization(object):
    def test_regularization(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            pixelization=pix.Rectangular,
            regularization=reg.Constant,
        )

        arguments = {
            galaxy_model.redshift.redshift: 2.0,
            galaxy_model.pixelization.shape_0: 24.0,
            galaxy_model.pixelization.shape_1: 23.0,
            galaxy_model.regularization.coefficient: 0.5,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.regularization.coefficient == 0.5

    def test_fixed_regularization(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            pixelization=pix.Voronoi(),
            regularization=reg.Constant(),
        )

        arguments = {galaxy_model.redshift.redshift: 2.0}

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.regularization.coefficient == 1.0

    def test__if_no_pixelization_raises_error(self):
        with pytest.raises(af.exc.PriorException):
            gp.GalaxyModel(redshift=g.Redshift, regularization=reg.Constant)


class TestHyperGalaxy(object):
    def test_hyper_galaxy(self,):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift, hyper_galaxy=g.HyperGalaxy)

        arguments = {
            galaxy_model.redshift.redshift: 0.2,
            galaxy_model.hyper_galaxy.contribution_factor: 1,
            galaxy_model.hyper_galaxy.noise_factor: 2,
            galaxy_model.hyper_galaxy.noise_power: 1.5,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.hyper_galaxy.contribution_factor == 1
        assert galaxy.hyper_galaxy.noise_factor == 2
        assert galaxy.hyper_galaxy.noise_power == 1.5

        assert galaxy.hyper_galaxy_image_1d is None

    def test_fixed_hyper_galaxy(self,):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift, hyper_galaxy=g.HyperGalaxy())

        arguments = {galaxy_model.redshift.redshift: 2.0}

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert galaxy.hyper_galaxy.contribution_factor == 0.0
        assert galaxy.hyper_galaxy.noise_factor == 0.0
        assert galaxy.hyper_galaxy.noise_power == 1.0

        assert galaxy.hyper_galaxy_image_1d is None


class TestFixedProfiles(object):
    def test_fixed_light(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift, light_profile=light_profiles.EllipticalSersic()
        )

        arguments = {galaxy_model.redshift.redshift: 2.0}

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert len(galaxy.light_profiles) == 1

    def test_fixed_mass(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift, nass_profile=mass_profiles.SphericalNFW()
        )

        arguments = {galaxy_model.redshift.redshift: 2.0}

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert len(galaxy.mass_profiles) == 1

    def test_fixed_and_variable(self):
        galaxy_model = gp.GalaxyModel(
            redshift=g.Redshift,
            mass_profile=mass_profiles.SphericalNFW(),
            light_profile=light_profiles.EllipticalSersic(),
            variable_light=light_profiles.EllipticalSersic,
        )

        arguments = {
            galaxy_model.redshift.redshift: 0.2,
            galaxy_model.variable_light.axis_ratio: 0.4,
            galaxy_model.variable_light.phi: 0.5,
            galaxy_model.variable_light.intensity.value: 0.6,
            galaxy_model.variable_light.effective_radius.value: 0.7,
            galaxy_model.variable_light.sersic_index: 0.8,
            galaxy_model.variable_light.centre.centre_0: 0,
            galaxy_model.variable_light.centre.centre_1: 0,
        }

        galaxy = galaxy_model.instance_for_arguments(arguments)

        assert len(galaxy.light_profiles) == 2
        assert len(galaxy.mass_profiles) == 1


class TestRedshift(object):
    def test_set_redshift_class(self):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift)
        galaxy_model.redshift = g.Redshift(3)
        assert galaxy_model.redshift == 3

    def test_set_redshift_float(self):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift)
        galaxy_model.redshift = 3
        # noinspection PyUnresolvedReferences
        assert galaxy_model.redshift == 3

    def test_set_redshift_constant(self):
        galaxy_model = gp.GalaxyModel(redshift=g.Redshift)
        galaxy_model.redshift = 3
        # noinspection PyUnresolvedReferences
        assert galaxy_model.redshift == 3


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy(
        redshift=3,
        sersic=light_profiles.EllipticalSersic(),
        exponential=light_profiles.EllipticalExponential(),
        spherical=mass_profiles.SphericalIsothermal(),
    )
