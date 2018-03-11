from auto_lens import prior
import pytest
from auto_lens.profiles import geometry_profiles, light_profiles, mass_profiles

import os
data_path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name='uniform_simple')
def make_uniform_simple():
    return prior.UniformPrior(lower_limit=0., upper_limit=1.)


@pytest.fixture(name='uniform_half')
def make_uniform_half():
    return prior.UniformPrior(lower_limit=0.5, upper_limit=1.)


class TestUniformPrior(object):
    def test__simple_assumptions(self, uniform_simple):
        assert uniform_simple.value_for(0.) == 0.
        assert uniform_simple.value_for(1.) == 1.
        assert uniform_simple.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self, uniform_half):
        assert uniform_half.value_for(0.) == 0.5
        assert uniform_half.value_for(1.) == 1.
        assert uniform_half.value_for(0.5) == 0.75


class MockClass(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockConfig(prior.Config):
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


class MockProfile(object):
    def __init__(self, centre=(0.0, 0.0), intensity=0.1):
        self.centre = centre
        self.intensity = intensity


class TestClassMappingCollection(object):
    def test__argument_extraction(self):
        collection = prior.ClassMap(MockConfig())
        collection.add_class("mock_class", MockClass)
        assert 1 == len(collection.prior_models)

        assert len(collection.priors_ordered_by_id) == 2

    def test_config_limits(self):
        collection = prior.ClassMap(MockConfig({"MockClass": {"one": ["u", 1., 2.]}}))

        collection.add_class("mock_class", MockClass)

        assert collection.mock_class.one.lower_limit == 1.
        assert collection.mock_class.one.upper_limit == 2.

    def test_config_prior_type(self):
        collection = prior.ClassMap(MockConfig({"MockClass": {"one": ["g", 1., 2.]}}))

        collection.add_class("mock_class", MockClass)

        assert isinstance(collection.mock_class.one, prior.GaussianPrior)

        assert collection.mock_class.one.mean == 1.
        assert collection.mock_class.one.sigma == 2.

    def test_attribution(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_class", MockClass)

        assert hasattr(collection, "mock_class")
        assert hasattr(collection.mock_class, "one")

    def test_tuple_arg(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        assert 3 == len(collection.priors_ordered_by_id)


class TestReconstruction(object):
    def test_simple_reconstruction(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_class", MockClass)

        reconstruction = collection.reconstruction_from_unit_vector([1., 1.])

        assert isinstance(reconstruction.mock_class, MockClass)
        assert reconstruction.mock_class.one == 1.
        assert reconstruction.mock_class.two == 1.

    def test_two_object_reconstruction(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_class_1", MockClass)
        collection.add_class("mock_class_2", MockClass)

        reconstruction = collection.reconstruction_from_unit_vector([1., 0., 0., 1.])

        assert isinstance(reconstruction.mock_class_1, MockClass)
        assert isinstance(reconstruction.mock_class_2, MockClass)

        assert reconstruction.mock_class_1.one == 1.
        assert reconstruction.mock_class_1.two == 0.

        assert reconstruction.mock_class_2.one == 0.
        assert reconstruction.mock_class_2.two == 1.

    def test_swapped_prior_construction(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_class_1", MockClass)
        collection.add_class("mock_class_2", MockClass)

        collection.mock_class_2.one = collection.mock_class_1.one

        reconstruction = collection.reconstruction_from_unit_vector([1., 0., 0.])

        assert isinstance(reconstruction.mock_class_1, MockClass)
        assert isinstance(reconstruction.mock_class_2, MockClass)

        assert reconstruction.mock_class_1.one == 1.
        assert reconstruction.mock_class_1.two == 0.

        assert reconstruction.mock_class_2.one == 1.
        assert reconstruction.mock_class_2.two == 0.

    def test_prior_replacement(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_class", MockClass)

        collection.mock_class.one = prior.UniformPrior(100, 200)

        reconstruction = collection.reconstruction_from_unit_vector([0., 0.])

        assert reconstruction.mock_class.one == 100.

    def test_tuple_arg(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        reconstruction = collection.reconstruction_from_unit_vector([1., 0., 0.])

        assert reconstruction.mock_profile.centre == (1., 0.)
        assert reconstruction.mock_profile.intensity == 0.

    def test_modify_tuple(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        collection.mock_profile.centre.centre_0 = prior.UniformPrior(1., 10.)

        reconstruction = collection.reconstruction_from_unit_vector([1., 1., 1.])

        assert reconstruction.mock_profile.centre == (10., 1.)

    def test_match_tuple(self):
        collection = prior.ClassMap(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        collection.mock_profile.centre.centre_1 = collection.mock_profile.centre.centre_0

        reconstruction = collection.reconstruction_from_unit_vector([1., 0.])

        assert reconstruction.mock_profile.centre == (1., 1.)
        assert reconstruction.mock_profile.intensity == 0.


class TestRealClasses(object):

    def test_combination(self):
        collection = prior.ClassMap(MockConfig(), source_light_profile=light_profiles.SersicLightProfile,
                                    lens_mass_profile=mass_profiles.CoredEllipticalIsothermalMassProfile,
                                    lens_light_profile=light_profiles.CoreSersicLightProfile)

        reconstruction = collection.reconstruction_from_unit_vector([1 for _ in range(len(collection.priors_ordered_by_id))])

        assert isinstance(reconstruction.source_light_profile, light_profiles.SersicLightProfile)
        assert isinstance(reconstruction.lens_mass_profile, mass_profiles.CoredEllipticalIsothermalMassProfile)
        assert isinstance(reconstruction.lens_light_profile, light_profiles.CoreSersicLightProfile)


class TestConfigFunctions:

    def test_loading_config(self):
        config = prior.Config(config_folder_path=data_path+"test_files/config")

        assert ['u', 0, 1.0] == config.get("geometry_profiles", "Profile", "centre_0")
        assert ['u', 0, 1.0] == config.get("geometry_profiles", "Profile", "centre_1")

    def test_reconstruction_from_unit_vector(self):
        collection = prior.ClassMap(prior.Config(config_folder_path=data_path+"test_files/config"), geometry_profile=geometry_profiles.Profile)

        reconstruction = collection.reconstruction_from_unit_vector([1., 1.])

        assert reconstruction.geometry_profile.centre == (1., 1.0)

    def test_reconstruction_from_physical_vector(self):
        collection = prior.ClassMap(prior.Config(config_folder_path=data_path+"test_files/config"), geometry_profile=geometry_profiles.Profile)

        reconstruction = collection.reconstruction_from_physical_vector([10., 50.])

        assert reconstruction.geometry_profile.centre == (10., 50.0)

    def test_inheritance(self):
        collection = prior.ClassMap(prior.Config(config_folder_path=data_path+"test_files/config"), geometry_profile=geometry_profiles.EllipticalProfile)

        reconstruction = collection.reconstruction_from_unit_vector([1., 1., 1., 1.])

        assert reconstruction.geometry_profile.centre == (1.0, 1.0)

    def test_true_config(self):
        collection = prior.ClassMap(elliptical_profile_1=geometry_profiles.EllipticalProfile,
                                    elliptical_profile_2=geometry_profiles.EllipticalProfile,
                                    spherical_profile=geometry_profiles.SphericalProfile,
                                    elliptical_light_profile=light_profiles.EllipticalLightProfile,
                                    sersic_light_profile=light_profiles.SersicLightProfile,
                                    exponential_light_profile=light_profiles.ExponentialLightProfile)

        reconstruction = collection.reconstruction_from_unit_vector([1 for _ in range(len(collection.priors_ordered_by_id))])

        assert isinstance(reconstruction.elliptical_profile_1, geometry_profiles.EllipticalProfile)
        assert isinstance(reconstruction.elliptical_profile_2, geometry_profiles.EllipticalProfile)
        assert isinstance(reconstruction.spherical_profile, geometry_profiles.SphericalProfile)

        assert isinstance(reconstruction.elliptical_light_profile, light_profiles.EllipticalLightProfile)
        assert isinstance(reconstruction.sersic_light_profile, light_profiles.SersicLightProfile)
        assert isinstance(reconstruction.exponential_light_profile, light_profiles.ExponentialLightProfile)


class TestHyperCube:

    def test__in_order_of_class_constructor_one_profile(self):
        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            geometry_profile=geometry_profiles.EllipticalProfile)

        assert collection.physical_vector_from_hypercube_vector([0.5, 0.5, 0.5, 0.5]) == [0.5, 0.5, 1.0, 1.0]

    def test__in_order_of_class_constructor_multiple_profiles(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        assert collection.physical_vector_from_hypercube_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == \
               [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]

    def test__in_order_of_class_constructor_multiple_profiles_bigger_range_of_unit_values(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        assert collection.physical_vector_from_hypercube_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) == \
               [0.1, 0.2, 0.6, 0.8, 0.5, 0.6, 0.7, 0.8, 1.8, 2.0]

    # TODO : Fix This - Also tuples and setting parameters equal to one another

    # def test__order_maintained_with_prior_change(self):
    #
    #     collection = prior.ClassMap(
    #         prior.Config(config_folder_path=data_path+"test_files/config"),
    #         profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
    #         profile_3=geometry_profiles.EllipticalProfile)
    #
    #     collection.profile_1.axis_ratio = prior.UniformPrior(100, 200)
    #
    #     assert collection.physical_vector_from_hypercube_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == \
    #            [0.5, 0.25, 150.0, 0.8, 0.5, 0.25, 0.5, 0.25, 0.75, 0.8]


class TestReconstructions:

    def test__in_order_of_class_constructor_one_profile(self):
        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile)

        reconstruction = collection.reconstruction_from_unit_vector([0.25, 0.5, 0.75, 1.0])

        assert reconstruction.profile_1.centre == (0.25, 0.5)
        assert reconstruction.profile_1.axis_ratio == 1.5
        assert reconstruction.profile_1.phi == 2.0

    def test__order_of_class_construtors_with_multiple_profiles(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        reconstruction = collection.reconstruction_from_unit_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert reconstruction.profile_1.centre == (0.1, 0.2)
        assert reconstruction.profile_1.axis_ratio == 0.6
        assert reconstruction.profile_1.phi == 0.8

        assert reconstruction.profile_2.centre == (0.5, 0.6)

        assert reconstruction.profile_3.centre == (0.7, 0.8)
        assert reconstruction.profile_3.axis_ratio == 1.8
        assert reconstruction.profile_3.phi == 2.0

    # TODO : The order of the parametes is not maintained when we change a prior - fix.

    # def test__in_order_of_class_constructor_order_maintained_with_prior_changes_simple_model(self):
    #
    #     collection = prior.ClassMap(
    #         prior.Config(config_folder_path=data_path+"test_files/config"),
    #         profile_1=geometry_profiles.EllipticalProfile)
    #
    #     collection.profile_1.centre.centre_1 = prior.UniformPrior(10, 20)
    #
    #     reconstruction = collection.reconstruction_from_unit_vector([0.1, 1.0, 0.3, 0.4])
    #
    #     assert reconstruction.profile_1.centre == (0.1, 20.0)
    #     assert reconstruction.profile_1.axis_ratio == 0.6
    #     assert reconstruction.profile_1.phi == 0.8

    # def test__in_order_of_class_constructor_order_maintained_with_prior_changes(self):
    #
    #     collection = prior.ClassMap(
    #         prior.Config(config_folder_path=data_path+"test_files/config"),
    #         profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
    #         profile_3=geometry_profiles.EllipticalProfile)
    #
    #     collection.profile_1.phi = prior.UniformPrior(100, 200)
    #     collection.profile_2.centre.centre_1 = prior.UniformPrior(10, 20)
    #
    #     reconstruction = collection.reconstruction_from_unit_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #
    #     assert reconstruction.profile_1.centre == (0.1, 0.2)
    #     assert reconstruction.profile_1.axis_ratio == 0.6
    #     assert reconstruction.profile_1.phi == 140.0
    #
    #     assert reconstruction.profile_2.centre == (0.5, 16.0)
    #
    #     assert reconstruction.profile_3.centre == (0.7, 0.8)
    #     assert reconstruction.profile_3.axis_ratio == 1.8
    #     assert reconstruction.profile_3.phi == 2.0

    # TODO : Same problem with a prior reassignment / pair

    # def test__in_order_of_class_constructor_order_maintained_when_prior_reassigned__simple_model(self):
    #
    #     collection = prior.ClassMap(
    #         prior.Config(config_folder_path=data_path+"test_files/config"),
    #         profile_1=geometry_profiles.EllipticalProfile)
    #
    #     collection.profile_1.centre.centre_1 = collection.profile_1.phi
    #
    #     reconstruction = collection.reconstruction_from_unit_vector([0.1, 1.0, 0.3, 0.4])
    #
    #     assert reconstruction.profile_1.centre == (0.1, 2.0)
    #     assert reconstruction.profile_1.axis_ratio == 0.6
    #     assert reconstruction.profile_1.phi == 0.8

    # TODO : This test works because we reasign each parameter in order... pretty useless.

    def test__check_order_for_different_unit_values(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        collection.profile_1.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.centre.centre_1 = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.axis_ratio = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.phi = prior.UniformPrior(0.0, 1.0)

        collection.profile_2.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_2.centre.centre_1 = prior.UniformPrior(0.0, 1.0)

        collection.profile_3.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.centre.centre_1 = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.axis_ratio = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.phi = prior.UniformPrior(0.0, 1.0)

        reconstruction = collection.reconstruction_from_unit_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert reconstruction.profile_1.centre == (0.1, 0.2)
        assert reconstruction.profile_1.axis_ratio == 0.3
        assert reconstruction.profile_1.phi == 0.4

        assert reconstruction.profile_2.centre == (0.5, 0.6)

        assert reconstruction.profile_3.centre == (0.7, 0.8)
        assert reconstruction.profile_3.axis_ratio == 0.9
        assert reconstruction.profile_3.phi == 1.0

    # TODO : It doesnt totally make sense to me why this one works tbh...

    def test__check_order_for_different_unit_values_and_set_priors_equal_to_one_another(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        collection.profile_1.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.centre.centre_1 = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.axis_ratio = prior.UniformPrior(0.0, 1.0)
        collection.profile_1.phi = prior.UniformPrior(0.0, 1.0)

        collection.profile_2.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_2.centre.centre_1 = prior.UniformPrior(0.0, 1.0)

        collection.profile_3.centre.centre_0 = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.centre.centre_1 = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.axis_ratio = prior.UniformPrior(0.0, 1.0)
        collection.profile_3.phi = prior.UniformPrior(0.0, 1.0)

        collection.profile_1.axis_ratio = collection.profile_1.phi
        collection.profile_3.centre.centre_1 = collection.profile_2.centre.centre_1

        reconstruction = collection.reconstruction_from_unit_vector([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        assert reconstruction.profile_1.centre == (0.2, 0.3)
        assert reconstruction.profile_1.axis_ratio == 0.4
        assert reconstruction.profile_1.phi == 0.4

        assert reconstruction.profile_2.centre == (0.5, 0.6)

        assert reconstruction.profile_3.centre == (0.7, 0.6)
        assert reconstruction.profile_3.axis_ratio == 0.8
        assert reconstruction.profile_3.phi == 0.9

    def test__check_order_for_physical_values(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile)

        reconstruction = collection.reconstruction_from_physical_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert reconstruction.profile_1.centre == (0.1, 0.2)
        assert reconstruction.profile_1.axis_ratio == 0.3
        assert reconstruction.profile_1.phi == 0.4

        assert reconstruction.profile_2.centre == (0.5, 0.6)

        assert reconstruction.profile_3.centre == (0.7, 0.8)
        assert reconstruction.profile_3.axis_ratio == 0.9
        assert reconstruction.profile_3.phi == 1.0


class TestMulitNestModels(object):

    def test__most_probable_reconstruction__simple_model(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile)

        most_probable = collection.reconstruction_most_probable(results_path=
                                                                   data_path+'test_files/multinest/short_')

        assert most_probable.profile_1.centre == (1.0, 2.0)
        assert most_probable.profile_1.axis_ratio == 3.0
        assert most_probable.profile_1.phi == 4.0

    def test__load_most_probable_from_multinest_weighted_sample__more_complex_model(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile, profile_4=geometry_profiles.Profile)

        most_probable = collection.reconstruction_most_probable(results_path=
                                                                 data_path+'test_files/multinest/long_')

        assert most_probable.profile_1.centre == (1.0, 2.0)
        assert most_probable.profile_1.axis_ratio == 3.0
        assert most_probable.profile_1.phi == 4.0

        assert most_probable.profile_2.centre == (-5.0, -6.0)

        assert most_probable.profile_3.centre == (-7.0, -8.0)
        assert most_probable.profile_3.axis_ratio == 9.0
        assert most_probable.profile_3.phi == 10.0

        assert most_probable.profile_4.centre == (11.0, 12.0)

    def test__load_most_likely_from_summary__simple_model(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile)

        most_likely = collection.reconstruction_most_likely(results_path=
                                                            data_path+'test_files/multinest/short_')

        assert most_likely.profile_1.centre == (5.0, 6.0)
        assert most_likely.profile_1.axis_ratio == 7.0
        assert most_likely.profile_1.phi == 8.0

    def test__load_most_likely_from_summary__more_complex_model(self):

        collection = prior.ClassMap(
            prior.Config(config_folder_path=data_path+"test_files/config"),
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.Profile,
            profile_3=geometry_profiles.EllipticalProfile, profile_4=geometry_profiles.Profile)

        most_likely = collection.reconstruction_most_likely(results_path=
                                                            data_path+'test_files/multinest/long_')

        assert most_likely.profile_1.centre == (13.0, 14.0)
        assert most_likely.profile_1.axis_ratio == 15.0
        assert most_likely.profile_1.phi == 16.0

        assert most_likely.profile_2.centre == (-17.0, -18.0)

        assert most_likely.profile_3.centre == (-19.0, -20.0)
        assert most_likely.profile_3.axis_ratio == 21.0
        assert most_likely.profile_3.phi == 22.0

        assert most_likely.profile_4.centre == (23.0, 24.0)


class TestGenerateParamNames(object):

    def test__input_class_map__single_profile__outputs_paramnames(self):

        collection = prior.ClassMap(prior.Config(config_folder_path=data_path + "test_files/config"),
        profile=geometry_profiles.EllipticalProfile)

        collection.output_paramnames_file(results_path=data_path + 'test_files/multinest/')

        paramnames_test = open('test_files/multinest/weighted_samples.paramnames')
        paramnames_str_1 = paramnames_test.readline()
        paramnames_str_2 = paramnames_test.readline()
        paramnames_str_3 = paramnames_test.readline()
        paramnames_str_4 = paramnames_test.readline()

        assert paramnames_str_1 == r'0_profile_centre_0                      $x$'+'\n'
        assert paramnames_str_2 == r'0_profile_centre_1                      $y$'+'\n'
        assert paramnames_str_3 == r'0_profile_axis_ratio                    $q$'+'\n'
        assert paramnames_str_4 == r'0_profile_phi                           $\phi$'+'\n'

    def test__input_class_map__two_profiles__outputs_paramnames(self):

        collection = prior.ClassMap(prior.Config(config_folder_path=data_path + "test_files/config"),
        profile=geometry_profiles.EllipticalProfile, mass_profile=mass_profiles.SphericalNFWMassProfile)

        collection.output_paramnames_file(results_path=data_path + 'test_files/multinest/')

        paramnames_test = open('test_files/multinest/weighted_samples.paramnames')
        paramnames_str_1 = paramnames_test.readline()
        paramnames_str_2 = paramnames_test.readline()
        paramnames_str_3 = paramnames_test.readline()
        paramnames_str_4 = paramnames_test.readline()
        paramnames_str_5 = paramnames_test.readline()
        paramnames_str_6 = paramnames_test.readline()
        paramnames_str_7 = paramnames_test.readline()
        paramnames_str_8 = paramnames_test.readline()

        assert paramnames_str_1 == r'0_profile_centre_0                      $x$'+'\n'
        assert paramnames_str_2 == r'0_profile_centre_1                      $y$'+'\n'
        assert paramnames_str_3 == r'0_profile_axis_ratio                    $q$'+'\n'
        assert paramnames_str_4 == r'0_profile_phi                           $\phi$'+'\n'
        assert paramnames_str_5 == r'1_mass_profile_centre_0                 $x_{\mathrm{d}}$'+'\n'
        assert paramnames_str_6 == r'1_mass_profile_centre_1                 $y_{\mathrm{d}}$'+'\n'
        assert paramnames_str_7 == r'1_mass_profile_kappa_s                  $\kappa_{\mathrm{d}}$'+'\n'
        assert paramnames_str_8 == r'1_mass_profile_scale_radius             $Rs_{\mathrm{d}}$'+'\n'

class TestUtility(object):

    def test_class_priors_dict(self):
        collection = prior.ClassMap(MockConfig(), mock_class=MockClass)

        assert list(collection.class_priors_dict.keys()) == ["mock_class"]
        assert len(collection.class_priors_dict["mock_class"]) == 2

        collection = prior.ClassMap(MockConfig(), mock_class_1=MockClass, mock_class_2=MockClass)

        collection.mock_class_1.one = collection.mock_class_2.one
        collection.mock_class_1.two = collection.mock_class_2.two

        assert collection.class_priors_dict["mock_class_1"] == collection.class_priors_dict["mock_class_2"]

    def test_value_vector_for_hypercube_vector(self):
        collection = prior.ClassMap(MockConfig(), mock_class=MockClass)

        collection.mock_class.two = prior.UniformPrior(upper_limit=100.)

        assert collection.physical_vector_from_hypercube_vector([1., 0.5]) == [1., 50.]
