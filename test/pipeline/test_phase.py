from src.pipeline import phase as ph
import pytest
from src.analysis import galaxy as g
from src.analysis import galaxy_prior as gp
from src.autopipe import non_linear


class MockResults(object):
    pass


class MockMaskedImage(object):
    pass


@pytest.fixture(name="phase")
def make_phase():
    return ph.SourceLensPhase(optimizer=non_linear.NonLinearOptimizer())


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior()


class TestPhase(object):
    def test_set_constants(self, phase, galaxy):
        phase.lens_galaxy = galaxy
        assert phase.optimizer.constant.lens_galaxy == galaxy
        assert not hasattr(phase.optimizer.variable, "lens_galaxy")

    def test_set_variables(self, phase, galaxy_prior):
        phase.lens_galaxy = galaxy_prior
        assert phase.optimizer.variable.lens_galaxy == galaxy_prior
        assert not hasattr(phase.optimizer.constant, "lens_galaxy")

    def test_run_arguments(self, phase):
        assert phase.last_results is None
        assert phase.masked_image is None
        results = MockResults()
        masked_image = MockMaskedImage()
        phase.run(masked_image, results)
        assert phase.last_results == results
        assert phase.masked_image == masked_image
