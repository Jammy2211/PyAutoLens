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


class ExtendedSourceLensPhase(ph.SourceLensPhase):
    def __init__(self, optimizer):
        super().__init__(optimizer)


@pytest.fixture(name="phase")
def make_phase():
    return ph.SourceLensPhase(optimizer=non_linear.NonLinearOptimizer())


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

    def test_run_arguments(self, phase, masked_image, results):
        assert phase.last_results is None
        assert phase.masked_image is None
        phase.run(masked_image=masked_image, last_results=results)
        assert phase.last_results == results
        assert phase.masked_image == masked_image
        assert phase.coords_collection is not None

    def test_default_arguments(self, phase, masked_image, results):
        assert phase.blurring_shape is None
        assert phase.sub_grid_size == 1
        phase.blurring_shape = (1, 1)
        assert phase.blurring_shape == (1, 1)
        phase.run(masked_image=masked_image, last_results=results)
        assert phase.blurring_shape == (1, 1)
        phase.blurring_shape = None
        assert phase.blurring_shape == (3, 3)
