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


class TestClassMappingCollection(object):
    def test__argument_extraction(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        collection.add_class(MockClass)
        assert 1 == len(collection.classes)
        assert 2 == len(collection.class_priors[0])

        assert len(collection) == 2

    def test__prior_substitution(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        uniform_prior = prior.UniformPrior("two")

        collection.add_class(MockClass, uniform_prior)

        assert uniform_prior is collection.class_priors[0][1]

        assert len(collection) == 2

    def test__prior_naming(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        collection.add_class(MockClass)
        collection.add_class(MockClass)

        assert "0.one" == collection.class_priors[0][0].path
        assert "0.two" == collection.class_priors[0][1].path

        assert "1.one" == collection.class_priors[1][0].path
        assert "1.two" == collection.class_priors[1][1].path

    def test_substitute_prior_naming(self):
        collection = prior.ClassMappingPriorCollection(MockConfig())
        priors = collection.add_class(MockClass)
        collection.add_class(MockClass, priors[0])

        assert "0.one" == collection.class_priors[0][0].path
        assert "0.two" == collection.class_priors[0][1].path

        assert "0.one" == collection.class_priors[1][0].path
        assert "1.two" == collection.class_priors[1][1].path

    def test_config_limits(self):
        collection = prior.ClassMappingPriorCollection(MockConfig({"MockClass": {"one": ["u", 1., 2.]}}))

        collection.add_class(MockClass)

        assert collection.class_priors[0][0].lower_limit == 1.
        assert collection.class_priors[0][0].upper_limit == 2.
