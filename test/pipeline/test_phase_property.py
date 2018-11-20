import os

import pytest
from autofit import conf
from autofit.core import model_mapper as mm
from autofit.core import non_linear

from autolens.model.galaxy import galaxy as g, galaxy_model as gp
from autolens.pipeline import phase as ph
from autolens.pipeline import phase_property

directory = os.path.dirname(os.path.realpath(__file__))

conf.instance = conf.Config("{}/../../workspace/config".format(directory),
                            "{}/../../workspace/output/".format(directory))


class NLO(non_linear.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector, constant):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = constant

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)
                for key, value in self.constant.__dict__.items():
                    setattr(instance, key, value)

                likelihood = analysis.fit(instance)
                self.result = non_linear.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.prior_count * [0.5])

        return fitness_function.result


@pytest.fixture(name='phase')
def make_phase():
    class MyPhase(ph.Phase):
        prop = phase_property.PhaseProperty("prop")

    return MyPhase(optimizer_class=NLO)


@pytest.fixture(name='list_phase')
def make_list_phase():
    class MyPhase(ph.Phase):
        prop = phase_property.PhasePropertyCollection("prop")

    return MyPhase(optimizer_class=NLO)


class TestPhaseProperty(object):
    def test_phase_property(self, phase):
        phase.prop = gp.GalaxyModel()

        assert phase.variable.prop == phase.prop

        galaxy = g.Galaxy()
        phase.prop = galaxy

        assert phase.constant.prop == galaxy
        assert not hasattr(phase.variable, "prop")

        phase.prop = gp.GalaxyModel()
        assert not hasattr(phase.constant, "prop")


class TestPhasePropertyList(object):
    def test_constants(self, list_phase):
        objects = [g.Galaxy(), g.Galaxy()]

        list_phase.prop = objects

        assert list_phase.constant.prop == objects
        assert len(list_phase.variable.prop) == 0

        assert list_phase.prop == objects

    def test_classes(self, list_phase):
        objects = [gp.GalaxyModel(), gp.GalaxyModel()]

        list_phase.prop = objects

        assert list_phase.variable.prop == objects
        assert len(list_phase.constant.prop) == 0

        assert list_phase.prop == objects

    def test_abstract_prior_models(self, list_phase):
        objects = [mm.AbstractPriorModel(), mm.AbstractPriorModel()]

        list_phase.prop = objects

        assert list_phase.variable.prop == objects
        assert len(list_phase.constant.prop) == 0

        assert list_phase.prop == objects

    def test_mix(self, list_phase):
        objects = [gp.GalaxyModel(), g.Galaxy()]

        list_phase.prop = objects

        assert list_phase.variable.prop == [objects[0]]
        assert list_phase.constant.prop == [objects[1]]

        assert list_phase.prop == objects

    def test_set_item(self, list_phase):
        galaxy_prior_0 = gp.GalaxyModel()
        objects = [galaxy_prior_0, g.Galaxy()]

        list_phase.prop = objects
        assert_ordered(list_phase.prop)

        galaxy_prior_1 = gp.GalaxyModel()
        list_phase.prop[1] = galaxy_prior_1

        assert_ordered(list_phase.prop)

        assert list_phase.constant.prop == []
        assert list_phase.variable.prop == [galaxy_prior_0, galaxy_prior_1]

        galaxy = g.Galaxy()

        list_phase.prop[0] = galaxy

        assert_ordered(list_phase.prop)

        assert list_phase.prop == [galaxy, galaxy_prior_1]

        assert list_phase.constant.prop == [galaxy]
        assert list_phase.variable.prop == [galaxy_prior_1]


class TestPhasePropertyCollectionAttributes(object):
    def test_set_list_as_dict(self, list_phase):
        galaxy_model = gp.GalaxyModel()
        list_phase.prop = dict(one=galaxy_model)

        assert len(list_phase.prop) == 1
        # noinspection PyUnresolvedReferences
        assert list_phase.prop.one == galaxy_model

    def test_override_property(self, list_phase):
        galaxy_model = gp.GalaxyModel()

        list_phase.prop = dict(one=gp.GalaxyModel())

        list_phase.prop.one = galaxy_model

        assert len(list_phase.prop) == 1
        assert list_phase.prop.one == galaxy_model

    def test_named_list_items(self, list_phase):
        galaxy_model = gp.GalaxyModel()
        list_phase.prop = [galaxy_model]

        # noinspection PyUnresolvedReferences
        assert list_phase.prop.prop_0 == galaxy_model

    def test_mix(self, list_phase):
        objects = dict(one=gp.GalaxyModel(), two=g.Galaxy())

        list_phase.prop = objects

        assert list_phase.variable.prop == [objects["one"]]
        assert list_phase.constant.prop == [objects["two"]]

        list_phase.prop.one = g.Galaxy()

        assert len(list_phase.variable.prop) == 0
        assert len(list_phase.constant.prop) == 2

    def test_named_attributes_in_variable(self, list_phase):
        galaxy_model = gp.GalaxyModel(variable_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert list_phase.variable.prior_count == 1
        assert list_phase.variable.one == galaxy_model

        instance = list_phase.variable.instance_from_prior_medians()

        assert instance.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_variable_override(self, list_phase):
        galaxy_model = gp.GalaxyModel(variable_redshift=True)
        list_phase.prop = dict(one=gp.GalaxyModel())

        assert list_phase.variable.prior_count == 0

        list_phase.prop.one = galaxy_model

        assert list_phase.variable.prior_count == 1
        assert list_phase.variable.one == galaxy_model

        instance = list_phase.variable.instance_from_prior_medians()

        assert instance.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_constant(self, list_phase):
        galaxy = g.Galaxy()
        list_phase.prop = dict(one=galaxy)

        assert list_phase.variable.prior_count == 0
        assert list_phase.constant.one == galaxy

    def test_singular_model_info(self, list_phase):
        galaxy_model = gp.GalaxyModel(variable_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert len(list_phase.variable.flat_prior_model_tuples) == 1
        assert len(list_phase.variable.info.split('\n')) == 4

    def test_shared_priors(self, list_phase):
        list_phase.prop = dict(one=gp.GalaxyModel(variable_redshift=True),
                               two=gp.GalaxyModel(variable_redshift=True))

        assert list_phase.variable.prior_count == 2

        # noinspection PyUnresolvedReferences
        list_phase.prop.one.redshift = list_phase.prop.two.redshift

        assert list_phase.variable.prior_count == 1

    def test_hasattr(self, list_phase):
        list_phase.prop = dict()

        assert not hasattr(list_phase.prop, "one")
        list_phase.prop = dict(one=gp.GalaxyModel(variable_redshift=True))

        assert hasattr(list_phase.prop, "one")


def assert_ordered(items):
    assert [n for n in range(len(items))] == [item.position for item in items]
