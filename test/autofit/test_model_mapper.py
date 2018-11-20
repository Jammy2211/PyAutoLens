import os

import pytest

import math

from autolens import conf
from autolens import exc
from autolens.autofit import model_mapper
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy, galaxy_model
from autolens.model.profiles import geometry_profiles, light_profiles, mass_profiles

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
                                             "test_files/configs/model_mapper/priors/default"))


@pytest.fixture(name='limit_config')
def make_limit_config():
    return conf.LimitConfig(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)),
                                             "test_files/configs/model_mapper/priors/limit"))


@pytest.fixture(name="width_config")
def make_width_config():
    return conf.WidthConfig(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)),
                                             "test_files/configs/model_mapper/priors/width"))


@pytest.fixture(name="initial_model")
def make_initial_model(test_config):
    return model_mapper.PriorModel(MockClassMM, test_config)


class MockClassGaussian(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockClassInf(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class TestPriorLimits(object):
    def test_in_or_out(self):
        prior = model_mapper.GaussianPrior(0, 1, 0, 1)
        with pytest.raises(exc.PriorLimitException):
            prior.assert_within_limits(-1)

        with pytest.raises(exc.PriorLimitException):
            prior.assert_within_limits(1.1)

        prior.assert_within_limits(0.)
        prior.assert_within_limits(0.5)
        prior.assert_within_limits(1.)

    def test_no_limits(self):
        prior = model_mapper.GaussianPrior(0, 1)

        prior.assert_within_limits(100)
        prior.assert_within_limits(-100)
        prior.assert_within_limits(0)
        prior.assert_within_limits(0.5)

    def test_uniform_prior(self):
        prior = model_mapper.UniformPrior(0, 1)

        with pytest.raises(exc.PriorLimitException):
            prior.assert_within_limits(-1)

        with pytest.raises(exc.PriorLimitException):
            prior.assert_within_limits(1.1)

        prior.assert_within_limits(0.)
        prior.assert_within_limits(0.5)
        prior.assert_within_limits(1.)

    def test_prior_creation(self, test_config, limit_config):
        mm = model_mapper.ModelMapper(test_config, limit_config=limit_config)
        mm.mock_class_gaussian = MockClassGaussian

        prior_tuples = mm.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.lower_limit == 0
        assert prior_tuples[0].prior.upper_limit == 1

        assert prior_tuples[1].prior.lower_limit == 0
        assert prior_tuples[1].prior.upper_limit == 2

    def test_out_of_limits(self, test_config, limit_config):
        mm = model_mapper.ModelMapper(test_config, limit_config=limit_config)
        mm.mock_class_gaussian = MockClassGaussian

        assert mm.instance_from_physical_vector([1, 2]) is not None

        with pytest.raises(exc.PriorLimitException):
            mm.instance_from_physical_vector(([1, 3]))

        with pytest.raises(exc.PriorLimitException):
            mm.instance_from_physical_vector(([-1, 2]))

    def test_inf(self, test_config, limit_config):
        mm = model_mapper.ModelMapper(test_config, limit_config=limit_config)
        mm.mock_class_inf = MockClassInf

        prior_tuples = mm.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.lower_limit == -math.inf
        assert prior_tuples[0].prior.upper_limit == 0

        assert prior_tuples[1].prior.lower_limit == 0
        assert prior_tuples[1].prior.upper_limit == math.inf

        assert mm.instance_from_physical_vector([-10000, 10000]) is not None

        with pytest.raises(exc.PriorLimitException):
            mm.instance_from_physical_vector(([1, 0]))

        with pytest.raises(exc.PriorLimitException):
            mm.instance_from_physical_vector(([0, -1]))


class TestPriorLinking(object):
    def test_same_class(self, initial_model):
        new_model = initial_model.linked_model_for_class(MockClassMM)

        assert new_model != initial_model
        assert new_model.one is initial_model.one
        assert new_model.two is initial_model.two

    def test_extended_class(self, initial_model):
        new_model = initial_model.linked_model_for_class(ExtendedMockClass)

        assert hasattr(new_model, "three")

    def test_override(self, initial_model):
        new_prior = model_mapper.GaussianPrior(1., 1.)
        new_model = initial_model.linked_model_for_class(MockClassMM, one=1., two=new_prior)

        assert new_model != initial_model
        assert new_model.one is not initial_model.one
        assert new_model.one == model_mapper.Constant(1.)
        assert isinstance(new_model.one, model_mapper.Constant)
        assert new_model.two is not initial_model.two
        assert new_model.two is new_prior

    def test_constants(self, initial_model):
        initial_model.one = 1

        new_model = initial_model.linked_model_for_class(MockClassMM)

        assert new_model.one == model_mapper.Constant(1)
        assert isinstance(new_model.one, model_mapper.Constant)
        assert new_model.one is initial_model.one
        assert new_model.two is initial_model.two

    def test_uniform_prior_mean(self):
        uniform_prior = model_mapper.UniformPrior(0., 1.)
        assert uniform_prior.mean == 0.5

        uniform_prior.mean = 1.
        assert uniform_prior.lower_limit == 0.5
        assert uniform_prior.upper_limit == 1.5

    def test_make_constants_variable(self, initial_model):
        initial_model.one = 1

        new_model = initial_model.linked_model_for_class(MockClassMM, make_constants_variable=True)

        assert new_model.one.mean == 0.5
        assert new_model.two is initial_model.two

    def test_tuple_passing(self, test_config):
        initial_model = model_mapper.PriorModel(MockProfile, test_config)
        initial_model.centre_0 = 1.
        assert isinstance(initial_model.centre_0, model_mapper.Constant)

        new_model = initial_model.linked_model_for_class(MockProfile)

        assert new_model.centre_0 is initial_model.centre_0
        assert new_model.centre_1 is initial_model.centre_1

    def test_is_tuple_like_attribute_name(self):
        assert model_mapper.is_tuple_like_attribute_name("centre_0")
        assert model_mapper.is_tuple_like_attribute_name("centre_1")
        assert not model_mapper.is_tuple_like_attribute_name("centre")
        assert model_mapper.is_tuple_like_attribute_name("centre_why_not_0")
        assert not model_mapper.is_tuple_like_attribute_name("centre_why_not")

    def test_tuple_name(self):
        assert model_mapper.tuple_name("centre_0") == "centre"
        assert model_mapper.tuple_name("centre_1") == "centre"
        assert model_mapper.tuple_name("centre_why_not_0") == "centre_why_not"


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

        assert three.prior_count == 14


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


class ExtendedMockClass(MockClassMM):
    def __init__(self, one, two, three):
        super().__init__(one, two)
        self.three = three


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

        model_info = mm.info

        assert model_info == """MockClassMM

mock_class_one                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
mock_class_two                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
"""

    def test_with_constant(self, test_config):
        mm = model_mapper.ModelMapper(test_config)
        mm.mock_class = MockClassMM

        mm.mock_class.two = model_mapper.Constant(1)

        model_info = mm.info

        assert model_info == """MockClassMM

mock_class_one                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
mock_class_two                                              Constant, value = 1
"""


class WithFloat(object):
    def __init__(self, value):
        self.value = value


class WithTuple(object):
    def __init__(self, tup=(0., 0.)):
        self.tup = tup


# noinspection PyUnresolvedReferences
class TestRegression(object):
    def test_tuple_constant_model_info(self, mapper):
        mapper.profile = light_profiles.EllipticalCoreSersic
        info = mapper.info

        mapper.profile.centre_0 = 1.

        assert len(mapper.profile.centre.constant_tuples) == 1
        assert len(mapper.profile.constant_tuples) == 1

        assert len(info.split('\n')) == len(mapper.info.split('\n'))

    def test_set_tuple_constant(self):
        mm = model_mapper.ModelMapper()
        mm.galaxy = galaxy_model.GalaxyModel(sersic=light_profiles.EllipticalSersic)

        assert mm.prior_count == 7

        mm.galaxy.sersic.centre_0 = model_mapper.Constant(0)
        mm.galaxy.sersic.centre_1 = model_mapper.Constant(0)

        assert mm.prior_count == 5

    def test_get_tuple_constants(self):
        mm = model_mapper.ModelMapper()
        mm.galaxy = galaxy_model.GalaxyModel(sersic=light_profiles.EllipticalSersic)

        assert isinstance(mm.galaxy.sersic.centre_0, model_mapper.Prior)
        assert isinstance(mm.galaxy.sersic.centre_1, model_mapper.Prior)

    def test_tuple_parameter(self, mapper):
        mapper.with_float = WithFloat
        mapper.with_tuple = WithTuple

        assert mapper.prior_count == 3

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 2

    def test_tuple_parameter_float(self, mapper):
        mapper.with_float = WithFloat
        mapper.with_tuple = WithTuple

        mapper.with_float.value = model_mapper.Constant(1)

        assert mapper.prior_count == 2

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 1

        instance = mapper.instance_from_unit_vector([0.])

        assert instance.with_float.value == 1
        assert instance.with_tuple.tup == (1., 0.)


class TestModelingMapper(object):
    def test__argument_extraction(self):
        mapper = model_mapper.ModelMapper(MockConfig())
        mapper.mock_class = MockClassMM
        assert 1 == len(mapper.prior_model_tuples)

        assert len(mapper.prior_tuples_ordered_by_id) == 2

    def test_config_limits(self):
        mapper = model_mapper.ModelMapper(MockConfig({"MockClassMM": {"one": ["u", 1., 2.]}}))

        mapper.mock_class = MockClassMM

        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.lower_limit == 1.
        # noinspection PyUnresolvedReferences
        assert mapper.mock_class.one.upper_limit == 2.

    def test_config_prior_type(self):
        mapper = model_mapper.ModelMapper(MockConfig({"MockClassMM": {"one": ["g", 1., 2.]}}),
                                          limit_config=MockConfig({"MockClassMM": {"one": [1., 2.]}}))

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

        assert 3 == len(mapper.prior_tuples_ordered_by_id)


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
        instance.other = galaxy_model.GalaxyModel()
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
        instance.galaxies = [galaxy_1, galaxy_2, galaxy_model.GalaxyModel]
        instance.galaxy_3 = g.Galaxy(redshift=3)
        instance.galaxy_prior = galaxy_model.GalaxyModel()

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
            [1 for _ in range(len(mapper.prior_tuples_ordered_by_id))])

        assert isinstance(model_map.source_light_profile, light_profiles.EllipticalSersic)
        assert isinstance(model_map.lens_mass_profile, mass_profiles.EllipticalCoredIsothermal)
        assert isinstance(model_map.lens_light_profile, light_profiles.EllipticalCoreSersic)

    def test_attribute(self):
        mm = model_mapper.ModelMapper(MockConfig())
        mm.cls_1 = MockClassMM

        assert 1 == len(mm.prior_model_tuples)
        assert isinstance(mm.cls_1, model_mapper.PriorModel)


class TestConfigFunctions:

    def test_loading_config(self, test_config):
        config = test_config

        assert ['u', 0, 1.0] == config.get("geometry_profiles", "GeometryProfile", "centre_0")
        assert ['u', 0, 1.0] == config.get("geometry_profiles", "GeometryProfile", "centre_1")

    def test_model_from_unit_vector(self, test_config, limit_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          limit_config=limit_config,
                                          geometry_profile=geometry_profiles.GeometryProfile)

        model_map = mapper.instance_from_unit_vector([1., 1.])

        assert model_map.geometry_profile.centre == (1., 1.0)

    def test_model_from_physical_vector(self, test_config, limit_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          limit_config=limit_config,
                                          geometry_profile=geometry_profiles.GeometryProfile)

        model_map = mapper.instance_from_physical_vector([1., 0.5])

        assert model_map.geometry_profile.centre == (1., 0.5)

    def test_inheritance(self, test_config, limit_config):
        mapper = model_mapper.ModelMapper(test_config,
                                          limit_config=limit_config,
                                          geometry_profile=geometry_profiles.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([1., 1., 1., 1.])

        assert model_map.geometry_profile.centre == (1.0, 1.0)

    def test_true_config(self, test_config, limit_config):
        config = test_config

        mapper = model_mapper.ModelMapper(config=config,
                                          limit_config=limit_config,
                                          sersic_light_profile=light_profiles.EllipticalSersic,
                                          elliptical_profile_1=geometry_profiles.EllipticalProfile,
                                          elliptical_profile_2=geometry_profiles.EllipticalProfile,
                                          spherical_profile=geometry_profiles.SphericalProfile,
                                          exponential_light_profile=light_profiles.EllipticalExponential)

        model_map = mapper.instance_from_unit_vector(
            [0.5 for _ in range(len(mapper.prior_tuples_ordered_by_id))])

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

        assert list(mapper.prior_model_name_prior_tuples_dict.keys()) == ["mock_class"]
        assert len(mapper.prior_model_name_prior_tuples_dict["mock_class"]) == 2

        mapper = model_mapper.ModelMapper(MockConfig(), mock_class_1=MockClassMM, mock_class_2=MockClassMM)

        mapper.mock_class_1.one = mapper.mock_class_2.one
        mapper.mock_class_1.two = mapper.mock_class_2.two

        assert mapper.prior_model_name_prior_tuples_dict["mock_class_1"] == mapper.prior_model_name_prior_tuples_dict[
            "mock_class_2"]

    def test_value_vector_for_hypercube_vector(self):
        mapper = model_mapper.ModelMapper(MockConfig(), mock_class=MockClassMM)

        mapper.mock_class.two = model_mapper.UniformPrior(upper_limit=100.)

        assert mapper.physical_values_ordered_by_class([1., 0.5]) == [1., 50.]

    def test_prior_prior_model_dict(self):
        mapper = model_mapper.ModelMapper(MockConfig(), mock_class=MockClassMM)

        assert len(mapper.prior_prior_model_dict) == 2
        assert mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[0][1]].cls == MockClassMM
        assert mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[1][1]].cls == MockClassMM


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

        assert isinstance(result.mock_class.centre.prior_tuples[0][1], model_mapper.GaussianPrior)
        assert isinstance(result.mock_class.centre.prior_tuples[1][1], model_mapper.GaussianPrior)
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

        instance = mapper.instance_from_physical_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2
        assert instance.two.one == 0.3
        assert instance.two.two == 0.4


class TestIndependentPriorModel(object):
    def test_associate_prior_model(self):
        prior_model = model_mapper.PriorModel(MockClassMM, MockConfig())

        mapper = model_mapper.ModelMapper(MockConfig())

        mapper.prior_model = prior_model

        assert len(mapper.prior_model_tuples) == 1

        instance = mapper.instance_from_physical_vector([0.1, 0.2])

        assert instance.prior_model.one == 0.1
        assert instance.prior_model.two == 0.2


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    return model_mapper.ListPriorModel(
        [model_mapper.PriorModel(MockClassMM, MockConfig()), model_mapper.PriorModel(MockClassMM, MockConfig())])


class TestListPriorModel(object):

    def test_instance_from_physical_vector(self, list_prior_model):
        mapper = model_mapper.ModelMapper(MockConfig())
        mapper.list = list_prior_model

        instance = mapper.instance_from_physical_vector([0.1, 0.2, 0.3, 0.4])

        assert isinstance(instance.list, list)
        assert len(instance.list) == 2
        assert instance.list[0].one == 0.1
        assert instance.list[0].two == 0.2
        assert instance.list[1].one == 0.3
        assert instance.list[1].two == 0.4

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

        assert len(mapper.prior_tuple_dict) == 1

    def test_retrieve_constants(self, mock_with_constant):
        assert len(mock_with_constant.constant_tuples) == 1

    def test_constant_prior_reconstruction(self, mock_with_constant):
        mapper = model_mapper.ModelMapper()
        mapper.mock_class = mock_with_constant

        instance = mapper.instance_for_arguments({mock_with_constant.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test_constant_in_config(self):
        mapper = model_mapper.ModelMapper()
        config = MockConfig({"MockClassMM": {"one": ["c", 3]}})

        mock_with_constant = model_mapper.PriorModel(MockClassMM, config)

        mapper.mock_class = mock_with_constant

        instance = mapper.instance_for_arguments({mock_with_constant.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test_constant_exchange(self, mock_with_constant, width_config):
        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.mock_class = mock_with_constant

        new_mapper = mapper.mapper_from_gaussian_means([1])

        assert len(new_mapper.mock_class.constant_tuples) == 1

    def test_set_float(self):
        prior_model = model_mapper.PriorModel(MockClassMM, MockConfig())
        prior_model.one = 3
        prior_model.two = 4.
        assert isinstance(prior_model.one, model_mapper.Constant)
        assert prior_model.one == model_mapper.Constant(3)
        assert prior_model.two == model_mapper.Constant(4.)

    def test_list_prior_model_constants(self, mapper):
        prior_model = model_mapper.PriorModel(MockClassMM, MockConfig())
        prior_model.one = 3
        prior_model.two = 4.
        assert isinstance(prior_model.one, model_mapper.Constant)
        mapper.mock_list = [prior_model]
        assert isinstance(mapper.mock_list, model_mapper.ListPriorModel)
        assert isinstance(prior_model.one, model_mapper.Constant)
        assert isinstance(mapper.mock_list[0].one, model_mapper.Constant)
        assert len(mapper.constant_tuples_ordered_by_id) == 2

    def test_set_for_tuple_prior(self):
        prior_model = model_mapper.PriorModel(light_profiles.EllipticalSersic, MockConfig())
        prior_model.centre_0 = 1.
        prior_model.centre_1 = 2.
        prior_model.axis_ratio = 1.
        prior_model.phi = 1.
        prior_model.intensity = 1.
        prior_model.effective_radius = 1.
        prior_model.sersic_index = 1.
        instance = prior_model.instance_for_arguments({})
        assert instance.centre == (1., 2.)


@pytest.fixture(name="mock_config")
def make_mock_config():
    return MockConfig()


@pytest.fixture(name="mapper")
def make_mapper(mock_config, width_config):
    return model_mapper.ModelMapper(config=mock_config, width_config=width_config)


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

        assert len(mapper.flat_prior_model_tuples) == 1

    def test_flatten_galaxy_prior_list(self, width_config):
        mapper = model_mapper.ModelMapper(width_config=width_config)
        mapper.list = [galaxy_model.GalaxyModel(variable_redshift=True)]

        assert len(mapper.flat_prior_model_tuples) == 1
        assert mapper.flat_prior_model_tuples[0][1].cls == galaxy.Redshift
