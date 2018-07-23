from src.pipeline import phase as ph
import pytest
from src.analysis import galaxy as g
from src.analysis import galaxy_prior as gp
from src.autopipe import non_linear


class MockPhase(ph.Phase):
    def __init__(self, optimizer):
        super(MockPhase, self).__init__(optimizer)


@pytest.fixture(name="phase")
def make_phase():
    return MockPhase(optimizer=non_linear.NonLinearOptimizer())


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior()


class TestPhase(object):
    def test_set_constants(self, phase, galaxy):
        phase.lens_galaxies = [galaxy]
        assert phase.optimizer.constant.lens_galaxies[0] == galaxy

    def test_set_variables(self, phase, galaxy_prior):
        phase.lens_galaxies = [galaxy_prior]
        assert phase.optimizer.variable.lens_galaxies[0] == galaxy_prior

    def test_set_constant_and_variable(self, phase, galaxy, galaxy_prior):
        phase.lens_galaxies = [galaxy, galaxy_prior]
        assert phase.optimizer.constant.lens_galaxies[0] == galaxy
        assert phase.optimizer.variable.lens_galaxies[0] == galaxy_prior
