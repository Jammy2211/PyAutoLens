from autolens.pipeline import phase as ph
from autolens.pipeline import phase_property
from autolens.analysis import galaxy as g
from autolens.analysis import galaxy_prior as gp
from autolens.autopipe import non_linear


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
        fitness_function(self.variable.total_parameters * [0.5])

        return fitness_function.result


class TestPhaseProperty(object):
    def test_phase_property(self):
        class MyPhase(ph.LensProfilePhase):
            prop = phase_property.phase_property("prop")

        phase = MyPhase(optimizer_class=NLO)

        phase.prop = gp.GalaxyPrior()

        assert phase.variable.prop == phase.prop

        galaxy = g.Galaxy()
        phase.prop = galaxy

        assert phase.constant.prop == galaxy
        assert not hasattr(phase.variable, "prop")

        phase.prop = gp.GalaxyPrior()
        assert not hasattr(phase.constant, "prop")
