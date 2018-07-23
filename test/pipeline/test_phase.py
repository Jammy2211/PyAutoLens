from src.pipeline import phase as ph
import pytest
from src.analysis import galaxy as g
from src.analysis import galaxy_prior as gp
from src.autopipe import non_linear
import numpy as np
from src.imaging import mask as msk


class MockResults(object):
    pass


class MockMaskedImage(object):
    def __init__(self, psf):
        self.psf = psf

    @property
    def mask(self):
        return msk.Mask.circular((10, 10), 1, 3)

    @property
    def image(self):
        return None


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

                likelihood = analysis.fit(**instance.__dict__)
                self.result = non_linear.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.total_parameters * [0.5])

        return fitness_function.result


@pytest.fixture(name="phase")
def make_phase():
    return ph.InitialSourceLensPhase(optimizer=NLO())


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior()


@pytest.fixture(name="masked_image")
def make_masked_image():
    return MockMaskedImage(np.zeros((3, 3)))


@pytest.fixture(name="results")
def make_results():
    return MockResults()


class TestPhase(object):
    def test_set_constants(self, phase, galaxy):
        phase.lens_galaxy = galaxy
        assert phase.optimizer.constant.lens_galaxy == galaxy
        assert not hasattr(phase.optimizer.variable, "lens_galaxy")

    def test_set_variables(self, phase, galaxy_prior):
        phase.lens_galaxy = galaxy_prior
        assert phase.optimizer.variable.lens_galaxy == galaxy_prior
        assert not hasattr(phase.optimizer.constant, "lens_galaxy")

    def test_default_arguments(self, phase, masked_image, results):
        assert phase.blurring_shape is None
        assert phase.sub_grid_size == 1
        phase.blurring_shape = (1, 1)
        assert phase.blurring_shape == (1, 1)
        phase.run(masked_image=masked_image, last_results=results)
        assert phase.blurring_shape == (1, 1)
        phase.blurring_shape = None
        assert phase.blurring_shape == (3, 3)
