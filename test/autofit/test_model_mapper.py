from autolens import conf
from autolens.autofit import model_mapper
import pytest
from autolens.profiles import geometry_profiles, light_profiles, mass_profiles
import os
from autolens.lensing import galaxy_prior
from autolens.lensing import galaxy as g

data_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='uniform_simple')
def make_uniform_simple():
    return model_mapper.UniformPrior(lower_limit=0., upper_limit=1.)


@pytest.fixture(name='uniform_half')
def make_uniform_half():
    return model_mapper.UniformPrior(lower_limit=0.5, upper_limit=1.)


@pytest.fixture(name='test_config')
def make_test_config():
    return conf.DefaultPriorConfig(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)),
                                             "test_files/config/priors/default"))


@pytest.fixture(name="width_config")
def make_width_config():
    return conf.WidthConfig(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)),
                                             "test_files/config/priors/width"))


class TestAddition(object):
    def test_abstract_plus_abstract(self):
        one = model_mapper.AbstractModel()
        two = model_mapper.AbstractModel()
        one.a = 'a'
        two.b = 'b'

        three = one + two

        assert three.a == 'a'
        assert three.b == 'b'

    def test_list_properties(self):
        one = model_mapper.AbstractModel()
        two = model_mapper.AbstractModel()
        one.a = ['a']
        two.a = ['b']

        three = one + two

        assert three.a == ['a', 'b']

    def test_instance_plus_instance(self):
        one = model_mapper.ModelInstance()
        two = model_mapper.ModelInstance()
        one.a = 'a'
        two.b = 'b'

        three = one + two

        assert three.a == 'a'
        assert three.b == 'b'

    def test_mapper_plus_mapper(self):
        one = model_mapper.ModelMapper()
        two = model_mapper.ModelMapper()
        one.a = model_mapper.PriorModel(light_profiles.EllipticalSersic)
        two.b = model_mapper.PriorModel(light_profiles.EllipticalSersic)

        three = one + two

        assert three.total_parameters == 14


class TestUniformPrior(object):
    def test__simple_assumptions(self, uniform_simple):
        assert uniform_simple.value_for(0.) == 0.
        assert uniform_simple.value_for(1.) == 1.
        assert uniform_simple.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self, uniform_half):
        assert uniform_half.value_for(0.) == 0.5
        assert uniform_half.value_for(1.) == 1.
        assert uniform_half.value_for(0.5) == 0.75


class MockClassMM(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockConfig(conf.DefaultPriorConfig):
    def __init__(self, d=None):
        super(MockConfig, self).__init__("")
        if d is not None:
            self.d = d
        else:
            self.d = {}

    def get_for_nearest_ancestor(self, cls, attribute_name):
        return self.get(None, cls.__name__, attribute_name)

    def get(self, _, class_name, var_name):
        try:
            return self.d[class_name][var_name]
        except KeyError:
            return ["u", 0, 1]


class MockWidthConfig(conf.WidthConfig):
    pass


class MockProfile(object):
    def __init__(self, centre=(0.0, 0.0), intensity=0.1):
        self.centre = centre
        self.intensity = intensity


class TestGenerateModelInfo(object):
    def test_basic(self, test_config):
        mm = model_mapper.ModelMapper(test_config)
        mm.mock_class = MockClassMM

        model_info = mm.model_info

        assert model_info == """VARIABLE:

MockClassMM

mock_class_one                          UniformPrior, lower_limit = 0.0, upper_limit = 1.0
mock_class_two                          UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""

    def test_with_constant(self, test_config):
        mm = model_mapper.ModelMapper(test_config)
        mm.mock_class = MockClassMM

        mm.mock_class.two = model_mapper.Constant(1)

        model_info = mm.model_info

        assert model_info == """VARIABLE:

MockClassMM

mock_class_one                          UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


class TestRegression(object):
    def test_set_tuple_constant(self):
        mm = model_mapper.ModelMapper()
        mm.galaxy = galaxy_prior.GalaxyPrior(sersic=light_profiles.EllipticalSersic)

        assert mm.total_parameters == 7

        mm.galaxy.sersic.centre_0 = model_mapper.Constant(0)
        mm.galaxy.sersic.centre_1 = model_mapper.Constant(0)

        assert mm.total_parameters == 5

    def test_get_tuple_constants(self):
        mm = model_mapper.ModelMapper()
        mm.galaxy = galaxy_prior.GalaxyPrior(sersic=light_profiles.EllipticalSersic)

        assert isinstance(mm.galaxy.sersic.centre_0, model_mapper.Prior)
        assert isinstance(mm.galaxy.sersic.centre_1, model_mapper.Prior)


class TestModelingMapper(object):
    def test__argument_extraction(self):
        mapper = model_mapper.ModelMapper(MockConfig())
        mapper.mock_class = MockClassMM
        assert 1 == len(mapper.prior_models)

        assert len(mapper.priors_ordered_by_id) == 2

    def test_config_limits(self):
        mapper = model_mapper.ModelMapper(MockConfig({"MockClassMM": {"one": ["u", 1., 2.]}}))

        mapper.mock_class = MockClassMM

        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.lower_limit == 1.
        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.upper_limit == 2.

    def test_config_prior_type(self):
        mapper = model_mapper.ModelMapper(MockConfig({"MockClassMM": {"one": ["g", 1., 2.]}}))

        mapper.mock_class = MockClassMM

        # noinspection PyUnresolvedReferences
        assert isinstance(mapper.mock_class.one, model_mapper.GaussianPrior)

        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.mean == 1.
        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.sigma == 2.

    def test_attribution(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_class = MockClassMM

        assert hasattr(mapper, "mock_class")
        assert hasattr(mapper.mock_class, "one")

    def test_tuple_arg(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_profile = MockProfile

        assert 3 == len(mapper.priors_ordered_by_id)


class TestModelInstance(object):
    def test_instances_of(self):
        instance = model_mapper.ModelInstance()
        instance.galaxy_1 = g.Galaxy()
        instance.galaxy_2 = g.Galaxy()
        assert instance.instances_of(g.Galaxy) == [instance.galaxy_1, instance.galaxy_2]

    def test_instances_of_filtering(self):
        instance = model_mapper.ModelInstance()
        instance.galaxy_1 = g.Galaxy()
        instance.galaxy_2 = g.Galaxy()
        instance.other = galaxy_prior.GalaxyPrior()
        assert instance.instances_of(g.Galaxy) == [instance.galaxy_1, instance.galaxy_2]

    def test_instances_from_list(self):
        instance = model_mapper.ModelInstance()
        galaxy_1 = g.Galaxy()
        galaxy_2 = g.Galaxy()
        instance.galaxies = [galaxy_1, galaxy_2]
        assert instance.instances_of(g.Galaxy) == [galaxy_1, galaxy_2]

    def test_non_trivial_instances_of(self):
        instance = model_mapper.ModelInstance()
        galaxy_1 = g.Galaxy(redshift=1)
        galaxy_2 = g.Galaxy(redshift=2)
        instance.galaxies = [galaxy_1, galaxy_2, galaxy_prior.GalaxyPrior]
        instance.galaxy_3 = g.Galaxy(redshift=3)
        instance.galaxy_prior = galaxy_prior.GalaxyPrior()

        assert instance.instances_of(g.Galaxy) == [instance.galaxy_3, galaxy_1, galaxy_2]

    def test_simple_model(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_class = MockClassMM

        model_map = mapper.instance_from_unit_vector([1., 1.])

        assert isinstance(model_map.mock_class, MockClassMM)
        assert model_map.mock_class.one == 1.
        assert model_map.mock_class.two == 1.

    def test_two_object_model(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_class_1 = MockClassMM
        mapper.mock_class_2 = MockClassMM

        model_map = mapper.instance_from_unit_vector([1., 0., 0., 1.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 0.
        assert model_map.mock_class_2.two == 1.

    def test_swapped_prior_construction(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_class_1 = MockClassMM
        mapper.mock_class_2 = MockClassMM

        # noinspection PyUnresolvedReferences
        mapper.mock_class_2.one = mapper.mock_class_1.one

        model_map = mapper.instance_from_unit_vector([1., 0., 0.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 1.
        assert model_map.mock_class_2.two == 0.

    def test_prior_replacement(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_class = MockClassMM

        mapper.mock_class.one = model_mapper.UniformPrior(100, 200)

        model_map = mapper.instance_from_unit_vector([0., 0.])

        assert model_map.mock_class.one == 100.

    def test_tuple_arg(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_profile = MockProfile

        model_map = mapper.instance_from_unit_vector([1., 0., 0.])

        assert model_map.mock_profile.centre == (1., 0.)
        assert model_map.mock_profile.intensity == 0.

    def test_modify_tuple(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_profile = MockProfile

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.centre.centre_0 = model_mapper.UniformPrior(1., 10.)

        model_map = mapper.instance_from_unit_vector([1., 1., 1.])

        assert model_map.mock_profile.centre == (10., 1.)

    def test_match_tuple(self):
        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.mock_profile = MockProfile

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.centre.centre_1 = mapper.mock_profile.centre.centre_0

        model_map = mapper.instance_from_unit_vector([1., 0.])

        assert model_map.mock_profile.centre == (1., 1.)
        assert model_map.mock_profile.intensity == 0.


class TestRealClasses(object):

    def test_combination(self):
        mapper = model_mapper.ModelMapper(MockConfig(),
                                          source_light_profile=light_profiles.EllipticalSersic,
                                          lens_mass_profile=mass_profiles.EllipticalCoredIsothermal,
                                          lens_light_profile=light_profiles.EllipticalCoreSersic)

        model_map = mapper.instance_from_unit_vector(
            [1 for _ in range(len(mapper.priors_ordered_by_id))])

        assert isinstance(model_map.source_light_profile, light_profiles.EllipticalSersic)
        assert isinstance(model_map.lens_mass_profile, mass_profiles.EllipticalCoredIsothermal)
        assert isinstance(model_map.lens_light_profile, light_profiles.EllipticalCoreSersic)

    def test_attribute(self):
        mm = model_mapper.ModelMapper(MockConfig())
        mm.cls_1 = MockClassMM

        assert 1 == len(mm.prior_models)
        assert isinstance(mm.cls_1, model_mapper.PriorModel)


class TestConfigFunctions:

    def test_loading_config(self, test_config):
        config = test_config

        assert ['u', 0, 1.0] == config.get("geometry_profiles", "GeometryProfile", "centre_0")
        assert ['u', 0, 1.0] == config.get("geometry_profiles", "GeometryProfile", "centre_1")

    def test_model_from_unit_vector(self, test_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          geometry_profile=geometry_profiles.GeometryProfile)

        model_map = mapper.instance_from_unit_vector([1., 1.])

        assert model_map.geometry_profile.centre == (1., 1.0)

    def test_model_from_physical_vector(self, test_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          geometry_profile=geometry_profiles.GeometryProfile)

        model_map = mapper.instance_from_physical_vector([10., 50.])

        assert model_map.geometry_profile.centre == (10., 50.0)

    def test_inheritance(self, test_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          geometry_profile=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([1., 1., 1., 1.])

        assert model_map.geometry_profile.centre == (1.0, 1.0)

    def test_true_config(self, test_config):
        config = test_config

        mapper = model_mapper.ModelMapper(config=config,
                                          sersic_light_profile=light_profiles.EllipticalSersic,
                                          elliptical_profile_1=geometry_profiles.EllipticalProfile,
                                          elliptical_profile_2=geometry_profiles.EllipticalProfile,
                                          spherical_profile=geometry_profiles.SphericalProfile,
                                          exponential_light_profile=light_profiles.EllipticalExponential)

        model_map = mapper.instance_from_unit_vector(
            [1 for _ in range(len(mapper.priors_ordered_by_id))])

        assert isinstance(model_map.elliptical_profile_1, geometry_profiles.EllipticalProfile)
        assert isinstance(model_map.elliptical_profile_2, geometry_profiles.EllipticalProfile)
        assert isinstance(model_map.spherical_profile, geometry_profiles.SphericalProfile)

        assert isinstance(model_map.sersic_light_profile, light_profiles.EllipticalSersic)
        assert isinstance(model_map.exponential_light_profile, light_profiles.EllipticalExponential)


class TestHyperCube:

    def test__in_order_of_class_constructor__one_profile(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            geometry_profile=geometry_profiles.EllipticalProfile)

        assert mapper.physical_values_ordered_by_class([0.5, 0.5, 0.5, 0.5]) == [1.0, 0.5, 0.5, 1.0]

    def test__in_order_of_class_constructor__multiple_profiles(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        unit_vector = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        assert mapper.physical_values_ordered_by_class(unit_vector) == [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5,
                                                                        1.0]

    def test__in_order_of_class_constructor__multiple_profiles_bigger_range_of_unit_values(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        unit_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        assert mapper.physical_values_ordered_by_class(unit_vector) == [0.6, 0.1, 0.2, 0.8, 0.5, 0.6, 1.8, 0.7, 0.8,
                                                                        2.0]

    def test__order_maintained_with_prior_change(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        unit_vector = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        before = mapper.physical_values_ordered_by_class(unit_vector)

        mapper.profile_1.axis_ratio = model_mapper.UniformPrior(0, 2)

        assert mapper.physical_values_ordered_by_class(unit_vector) == before


class TestModelInstancesRealClasses(object):

    def test__in_order_of_class_constructor__one_profile(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([0.25, 0.5, 0.75, 1.0])

        assert model_map.profile_1.centre == (0.25, 0.5)
        assert model_map.profile_1.axis_ratio == 1.5
        assert model_map.profile_1.phi == 2.0

    def test__in_order_of_class_constructor___multiple_profiles(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.6
        assert model_map.profile_1.phi == 0.8

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 1.8
        assert model_map.profile_3.phi == 2.0

    def test__check_order_for_different_unit_values(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        mapper.profile_1.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.axis_ratio = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.phi = model_mapper.UniformPrior(0.0, 1.0)

        mapper.profile_2.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_2.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)

        mapper.profile_3.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.axis_ratio = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.phi = model_mapper.UniformPrior(0.0, 1.0)

        model_map = mapper.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.3
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 0.9
        assert model_map.profile_3.phi == 1.0

    def test__check_order_for_different_unit_values_and_set_priors_equal_to_one_another(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        mapper.profile_1.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.axis_ratio = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_1.phi = model_mapper.UniformPrior(0.0, 1.0)

        mapper.profile_2.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_2.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)

        mapper.profile_3.centre.centre_0 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.centre.centre_1 = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.axis_ratio = model_mapper.UniformPrior(0.0, 1.0)
        mapper.profile_3.phi = model_mapper.UniformPrior(0.0, 1.0)

        mapper.profile_1.axis_ratio = mapper.profile_1.phi
        mapper.profile_3.centre.centre_1 = mapper.profile_2.centre.centre_1

        model_map = mapper.instance_from_unit_vector([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        assert model_map.profile_1.centre == (0.2, 0.3)
        assert model_map.profile_1.axis_ratio == 0.4
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.6)
        assert model_map.profile_3.axis_ratio == 0.8
        assert model_map.profile_3.phi == 0.9

    def test__check_order_for_physical_values(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_physical_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.3
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 0.9
        assert model_map.profile_3.phi == 1.0

    def test__from_prior_medians__one_model(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5])

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

    def test__from_prior_medians__multiple_models(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.GeometryProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

        assert model_map.profile_2.centre == model_2.profile_2.centre == (0.5, 0.5)

        assert model_map.profile_3.centre == model_2.profile_3.centre == (0.5, 0.5)
        assert model_map.profile_3.axis_ratio == model_2.profile_3.axis_ratio == 1.0
        assert model_map.profile_3.phi == model_2.profile_3.phi == 1.0

    def test__from_prior_medians__one_model__set_one_parameter_to_another(self, test_config):
        mapper = model_mapper.ModelMapper(
            test_config,
            profile_1=geometry_profiles.EllipticalProfile)

        mapper.profile_1.axis_ratio = mapper.profile_1.phi

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5])

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

    def test_physical_vector_from_prior_medians(self, test_config):
        mapper = model_mapper.ModelMapper()
        mapper.mock_class = model_mapper.PriorModel(MockClassMM, test_config)

        assert mapper.physical_values_from_prior_medians == [0.5, 0.5]


class TestUtility(object):

    def test_class_priors_dict(self):
        mapper = model_mapper.ModelMapper(MockConfig(), mock_class=MockClassMM)

        assert list(mapper.class_priors_dict.keys()) == ["mock_class"]
        assert len(mapper.class_priors_dict["mock_class"]) == 2

        mapper = model_mapper.ModelMapper(MockConfig(), mock_class_1=MockClassMM, mock_class_2=MockClassMM)

        mapper.mock_class_1.one = mapper.mock_class_2.one
        mapper.mock_class_1.two = mapper.mock_class_2.two

        assert mapper.class_priors_dict["mock_class_1"] == mapper.class_priors_dict["mock_class_2"]

    def test_value_vector_for_hypercube_vector(self):
        mapper = model_mapper.ModelMapper(MockConfig(), mock_class=MockClassMM)

        mapper.mock_class.two = model_mapper.UniformPrior(upper_limit=100.)

        assert mapper.physical_values_ordered_by_class([1., 0.5]) == [1., 50.]

    def test_prior_prior_model_dict(self):
        mapper = model_mapper.ModelMapper(MockConfig(), mock_class=MockClassMM)

        assert len(mapper.prior_prior_model_dict) == 2
        assert mapper.prior_prior_model_dict[mapper.priors_ordered_by_id[0][1]].cls == MockClassMM
        assert mapper.prior_prior_model_dict[mapper.priors_ordered_by_id[1][1]].cls == MockClassMM


class TestPriorReplacement(object):

    def test_prior_replacement(self, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config=width_config, mock_class=MockClassMM)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, model_mapper.GaussianPrior)

    def test_replace_priors_with_gaussians_from_tuples(self, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config=width_config, mock_class=MockClassMM)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, model_mapper.GaussianPrior)

    def test_replacing_priors_for_profile(self, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config=width_config, mock_class=MockProfile)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3), (5, 3)])

        assert isinstance(result.mock_class.centre.priors[0][1], model_mapper.GaussianPrior)
        assert isinstance(result.mock_class.centre.priors[1][1], model_mapper.GaussianPrior)
        assert isinstance(result.mock_class.intensity, model_mapper.GaussianPrior)

    def test_replace_priors_for_two_classes(self, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config=width_config, one=MockClassMM, two=MockClassMM)

        result = mapper.mapper_from_gaussian_tuples([(1, 1), (2, 1), (3, 1), (4, 1)])

        assert result.one.one.mean == 1
        assert result.one.two.mean == 2
        assert result.two.one.mean == 3
        assert result.two.two.mean == 4


class TestArguments(object):
    def test_same_argument_name(self, test_config):
        mapper = model_mapper.ModelMapper()

        mapper.one = model_mapper.PriorModel(MockClassMM, test_config)
        mapper.two = model_mapper.PriorModel(MockClassMM, test_config)

        instance = mapper.instance_from_physical_vector([1, 2, 3, 4])

        assert instance.one.one == 1
        assert instance.one.two == 2
        assert instance.two.one == 3
        assert instance.two.two == 4


class TestIndependentPriorModel(object):
    def test_associate_prior_model(self):
        prior_model = model_mapper.PriorModel(MockClassMM, MockConfig())

        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.prior_model = prior_model

        assert len(mapper.prior_models) == 1

        instance = mapper.instance_from_physical_vector([1, 2])

        assert instance.prior_model.one == 1
        assert instance.prior_model.two == 2


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    return model_mapper.ListPriorModel(
        [model_mapper.PriorModel(MockClassMM, MockConfig()), model_mapper.PriorModel(MockClassMM, MockConfig())])


class TestListPriorModel(object):

    def test_instance_from_physical_vector(self, list_prior_model):
        mapper = model_mapper.ModelMapper(MockConfig())
        mapper.list = list_prior_model

        instance = mapper.instance_from_physical_vector([1, 2, 3, 4])

        assert isinstance(instance.list, list)
        assert len(instance.list) == 2
        assert instance.list[0].one == 1
        assert instance.list[0].two == 2
        assert instance.list[1].one == 3
        assert instance.list[1].two == 4

    def test_prior_results_for_gaussian_tuples(self, list_prior_model, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config)
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples([(1, 5), (2, 5), (3, 5), (4, 5)])

        assert isinstance(gaussian_mapper.list, list)
        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 5
        assert gaussian_mapper.list[0].two.sigma == 5
        assert gaussian_mapper.list[1].one.sigma == 5
        assert gaussian_mapper.list[1].two.sigma == 5

    def test_prior_results_for_gaussian_tuples__include_override_from_width_file(self, list_prior_model, width_config):
        mapper = model_mapper.ModelMapper(MockConfig(), width_config)
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples([(1, 0), (2, 0), (3, 0), (4, 0)])

        assert isinstance(gaussian_mapper.list, list)
        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2

    def test_automatic_boxing(self):
        mapper = model_mapper.ModelMapper(MockConfig())
        mapper.list = [model_mapper.PriorModel(MockClassMM, MockConfig()),
                       model_mapper.PriorModel(MockClassMM, MockConfig())]

        assert isinstance(mapper.list, model_mapper.ListPriorModel)


@pytest.fixture(name="mock_with_constant")
def make_mock_with_constant():
    mock_with_constant = model_mapper.PriorModel(MockClassMM, MockConfig())
    mock_with_constant.one = model_mapper.Constant(3)
    return mock_with_constant


class TestConstant(object):
    def test_constant_prior_count(self, mock_with_constant):
        mapper = model_mapper.ModelMapper()
        mapper.mock_class = mock_with_constant

        assert len(mapper.prior_set) == 1

    def test_retrieve_constants(self, mock_with_constant):
        assert len(mock_with_constant.constants) == 1

    def test_constant_prior_reconstruction(self, mock_with_constant):
        mapper = model_mapper.ModelMapper()
        mapper.mock_class = mock_with_constant

        instance = mapper.instance_from_arguments({mock_with_constant.two: 5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 5

    def test_constant_in_config(self):
        mapper = model_mapper.ModelMapper()
        config = MockConfig({"MockClassMM": {"one": ["c", 3]}})

        mock_with_constant = model_mapper.PriorModel(MockClassMM, config)

        mapper.mock_class = mock_with_constant

        instance = mapper.instance_from_arguments({mock_with_constant.two: 5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 5

    def test_constant_exchange(self, mock_with_constant, width_config):
        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.mock_class = mock_with_constant

        new_mapper = mapper.mapper_from_gaussian_means([1])

        assert len(new_mapper.mock_class.constants) == 1


@pytest.fixture(name="mapper_with_one")
def make_mapper_with_one(test_config, width_config):
    mapper = model_mapper.ModelMapper(width_config=width_config)
    mapper.one = model_mapper.PriorModel(MockClassMM, config=test_config)
    return mapper


@pytest.fixture(name="mapper_with_list")
def make_mapper_with_list(test_config, width_config):
    mapper = model_mapper.ModelMapper(width_config=width_config)
    mapper.list = [model_mapper.PriorModel(MockClassMM, config=test_config),
                   model_mapper.PriorModel(MockClassMM, config=test_config)]
    return mapper


class TestGaussianWidthConfig(object):

    def test_config(self, width_config):
        assert 1 == width_config.get('test_model_mapper', 'MockClassMM', 'one')
        assert 2 == width_config.get('test_model_mapper', 'MockClassMM', 'two')

    def test_prior_classes(self, mapper_with_one):
        assert mapper_with_one.prior_class_dict == {mapper_with_one.one.one: MockClassMM,
                                                    mapper_with_one.one.two: MockClassMM}

    def test_prior_classes_list(self, mapper_with_list):
        assert mapper_with_list.prior_class_dict == {mapper_with_list.list[0].one: MockClassMM,
                                                     mapper_with_list.list[0].two: MockClassMM,
                                                     mapper_with_list.list[1].one: MockClassMM,
                                                     mapper_with_list.list[1].two: MockClassMM}

    def test_basic_gaussian_for_mean(self, mapper_with_one):
        gaussian_mapper = mapper_with_one.mapper_from_gaussian_means([3, 4])

        assert gaussian_mapper.one.one.sigma == 1
        assert gaussian_mapper.one.two.sigma == 2
        assert gaussian_mapper.one.one.mean == 3
        assert gaussian_mapper.one.two.mean == 4

    def test_gaussian_mean_for_list(self, mapper_with_list):
        gaussian_mapper = mapper_with_list.mapper_from_gaussian_means([3, 4, 5, 6])

        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2
        assert gaussian_mapper.list[0].one.mean == 3
        assert gaussian_mapper.list[0].two.mean == 4
        assert gaussian_mapper.list[1].one.mean == 5
        assert gaussian_mapper.list[1].two.mean == 6

    def test_gaussian_for_mean(self, test_config, width_config):
        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.one = model_mapper.PriorModel(MockClassMM, config=test_config)
        mapper.two = model_mapper.PriorModel(MockClassMM, config=test_config)

        gaussian_mapper = mapper.mapper_from_gaussian_means([3, 4, 5, 6])

        assert gaussian_mapper.one.one.sigma == 1
        assert gaussian_mapper.one.two.sigma == 2
        assert gaussian_mapper.two.one.sigma == 1
        assert gaussian_mapper.two.two.sigma == 2
        assert gaussian_mapper.one.one.mean == 3
        assert gaussian_mapper.one.two.mean == 4
        assert gaussian_mapper.two.one.mean == 5
        assert gaussian_mapper.two.two.mean == 6

    def test_no_override(self, test_config):
        mapper = model_mapper.ModelMapper()

        mapper.one = model_mapper.PriorModel(MockClassMM, config=test_config)

        model_mapper.ModelMapper()

        assert mapper.one is not None


class TestFlatPriorModel(object):
    def test_flatten_list(self, width_config, test_config):
        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.list = [model_mapper.PriorModel(MockClassMM, config=test_config)]

        assert len(mapper.flat_prior_models) == 1

    def test_flatten_galaxy_prior_list(self, width_config):
        from autolens.lensing import galaxy

        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.list = [galaxy_prior.GalaxyPrior(variable_redshift=True)]

        assert len(mapper.flat_prior_models) == 1
        assert mapper.flat_prior_models[0][1].cls == galaxy.Redshift
