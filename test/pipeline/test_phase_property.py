from autolens.pipeline import phase as ph
from autolens.pipeline import phase_property
from autolens.lensing import galaxy as g
from autolens.lensing import galaxy_model as gp
from autolens.autofit import non_linear
from autolens.autofit import model_mapper as mm
import pytest


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
        fitness_function(self.variable.total_priors * [0.5])

        return fitness_function.result


@pytest.fixture(name='phase')
def make_phase():
    class MyPhase(ph.Phase):
        prop = phase_property.PhaseProperty("prop")

    return MyPhase(optimizer_class=NLO)


@pytest.fixture(name='list_phase')
def make_list_phase():
    class MyPhase(ph.Phase):
        prop = phase_property.PhasePropertyList("prop")

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


def assert_ordered(items):
    assert [n for n in range(len(items))] == [item.position for item in items]
