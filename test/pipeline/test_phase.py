from src.pipeline import phase as ph
import pytest
from src.analysis import galaxy as g
from src.analysis import galaxy_prior as gp
from src.autopipe import non_linear
import numpy as np
from src.imaging import mask as msk
from src.imaging import image as img
from src.imaging import masked_image as mi


class MockResults(object):
    pass


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
    shape = (10, 10)
    image = img.Image(np.array(np.zeros(shape)), psf=img.PSF(np.ones((3, 3)), 1), background_noise=np.ones(shape))
    mask = msk.Mask.circular(shape, 1, 3)
    return mi.MaskedImage(image, mask)


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

    def test_default_arguments(self, phase, masked_image, results, galaxy_prior):
        phase.lens_galaxy = galaxy_prior
        phase.source_galaxy = galaxy_prior
        assert phase.blurring_shape is None
        assert phase.sub_grid_size == 1
        phase.blurring_shape = (1, 1)
        assert phase.blurring_shape == (1, 1)
        phase.run(masked_image=masked_image, last_results=results)
        assert phase.blurring_shape == (1, 1)

    def test_mask_analysis(self, phase):
        phase.make_analysis()
