import prior
import pytest


@pytest.fixture(name='uniform_simple')
def make_uniform_simple():
    return prior.UniformPrior("one", lower_limit=0., upper_limit=1.)


@pytest.fixture(name='uniform_half')
def make_uniform_half():
    return prior.UniformPrior("two", lower_limit=0.5, upper_limit=1.)


@pytest.fixture(name='collection')
def make_collection(uniform_simple, uniform_half):
    return prior.PriorCollection(uniform_simple, uniform_half)


class TestUniformPrior(object):
    def test__simple_assumptions(self, uniform_simple):
        assert uniform_simple.value_for(0.) == 0.
        assert uniform_simple.value_for(1.) == 1.
        assert uniform_simple.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self, uniform_half):
        assert uniform_half.value_for(0.) == 0.5
        assert uniform_half.value_for(1.) == 1.
        assert uniform_half.value_for(0.5) == 0.75

    def test__argument(self, uniform_simple):
        assert uniform_simple.argument_for(0.) == ("one", 0)
        assert uniform_simple.argument_for(0.5) == ("one", 0.5)


class MockClass(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class TestCollection(object):
    def test__arguments(self, collection):
        assert collection.arguments_for_vector([0., 0.]) == {"one": 0., "two": 0.5}
        assert collection.arguments_for_vector([1., 0.]) == {"one": 1., "two": 0.5}

    def test__equals(self, uniform_simple):
        assert uniform_simple != prior.Prior("two")
        assert uniform_simple == prior.Prior("one")

    def test__override(self, collection):
        collection.add(prior.UniformPrior("one", lower_limit=1., upper_limit=2.))

        assert len(collection) == 2
        assert collection[0].lower_limit == 1.
        assert collection[0].name == "one"

    def test__exceptions(self, collection, uniform_simple):
        with pytest.raises(AssertionError):
            collection.arguments_for_vector([0])

        with pytest.raises(AssertionError):
            collection.append(uniform_simple)

    def test__construct(self, collection):
        mock_object = MockClass(**collection.arguments_for_vector([1, 1]))
        assert mock_object.one == 1.
        assert mock_object.two == 1.


class MockConfig(object):
    def __init__(self, d=None):
        if d is not None:
            self.d = d
        else:
            self.d = {}

    def get(self, class_name, var_name):
        try:
            return self.d[class_name][var_name]
        except KeyError:
            return ["u", 0, 1]


class MockProfile(object):
    def __init__(self, intensity, centre=(0, 0)):
        self.intensity = intensity
        self.centre = centre


class TestClassMappingCollection(object):
    def test__argument_extraction(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        collection.add_class("mock_class", MockClass)
        assert 1 == len(collection.prior_models)

        assert len(collection.priors) == 2

    def test__prior_naming(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        collection.add_class("mock_class_1", MockClass)
        collection.add_class("mock_class_2", MockClass)

        assert "0.one" == collection.mock_class_1.one.path
        assert "0.two" == collection.mock_class_1.two.path

        assert "1.one" == collection.mock_class_2.one.path
        assert "1.two" == collection.mock_class_2.two.path

    def test_config_limits(self):
        collection = prior.ClassMappingPriorCollection(MockConfig({"MockClass": {"one": ["u", 1., 2.]}}))

        collection.add_class("mock_class", MockClass)

        assert collection.mock_class.one.lower_limit == 1.
        assert collection.mock_class.one.upper_limit == 2.

    def test_config_prior_type(self):
        collection = prior.ClassMappingPriorCollection(MockConfig({"MockClass": {"one": ["g", 1., 2.]}}))

        collection.add_class("mock_class", MockClass)

        assert isinstance(collection.mock_class.one, prior.GaussianPrior)

        assert collection.mock_class.one.mean == 1.
        assert collection.mock_class.one.sigma == 2.

    def test_attribution(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_class", MockClass)

        assert hasattr(collection, "mock_class")
        assert hasattr(collection.mock_class, "one")

    def test_tuple_arg(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        assert 3 == len(collection.priors)


class TestReconstruction(object):
    def test_simple_reconstruction(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_class", MockClass)

        reconstruction = collection.reconstruction_for_vector([1., 1.])

        assert isinstance(reconstruction.mock_class, MockClass)
        assert reconstruction.mock_class.one == 1.
        assert reconstruction.mock_class.two == 1.

    def test_two_object_reconstruction(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_class_1", MockClass)
        collection.add_class("mock_class_2", MockClass)

        reconstruction = collection.reconstruction_for_vector([1., 0., 0., 1.])

        assert isinstance(reconstruction.mock_class_1, MockClass)
        assert isinstance(reconstruction.mock_class_2, MockClass)

        assert reconstruction.mock_class_1.one == 1.
        assert reconstruction.mock_class_1.two == 0.

        assert reconstruction.mock_class_2.one == 0.
        assert reconstruction.mock_class_2.two == 1.

    def test_swapped_prior_construction(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_class_1", MockClass)
        collection.add_class("mock_class_2", MockClass)

        collection.mock_class_2.one = collection.mock_class_1.one

        reconstruction = collection.reconstruction_for_vector([1., 0., 0.])

        assert isinstance(reconstruction.mock_class_1, MockClass)
        assert isinstance(reconstruction.mock_class_2, MockClass)

        assert reconstruction.mock_class_1.one == 1.
        assert reconstruction.mock_class_1.two == 0.

        assert reconstruction.mock_class_2.one == 1.
        assert reconstruction.mock_class_2.two == 0.

    def test_prior_replacement(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_class", MockClass)

        collection.mock_class.one = prior.UniformPrior("one", 100, 200)

        reconstruction = collection.reconstruction_for_vector([0., 0.])

        assert reconstruction.mock_class.one == 100.

    def test_tuple_arg(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())

        collection.add_class("mock_profile", MockProfile)

        reconstruction = collection.reconstruction_for_vector([1., 0., 0.])

        assert reconstruction.mock_profile.intensity == 0.
        assert reconstruction.mock_profile.centre == (1., 0.)
