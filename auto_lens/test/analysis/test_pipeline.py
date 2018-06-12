from auto_lens.analysis import pipeline as pl
from auto_lens.analysis import galaxy_prior
from auto_lens.analysis import model_mapper as mm
import pytest
import os
import numpy as np


class MockResult:
    pass


class MockStage:
    def __init__(self):
        self.is_run = False

    def run(self):
        self.is_run = True
        return MockResult()


class MockImage:
    pass


class MockPixelization:
    pass


class MockPrior:
    pass


class MockPriorModel:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.centre = "centre for {}".format(name)
        self.phi = "phi for {}".format(name)


class MockModelInstance:
    pass


class MockNLO:
    def __init__(self):
        self.priors = None
        self.fitness_function = None

    def run(self, fitness_function, priors):
        self.fitness_function = fitness_function
        self.priors = priors
        fitness_function([0.5, 0.5])


class MockMask:
    # noinspection PyMethodMayBeStatic
    def compute_grid_coords_image(self):
        return np.array([[-1., -1.], [1., 1.]])


@pytest.fixture(name='test_config')
def make_test_config():
    return mm.Config(
        config_folder_path="{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/config"))


@pytest.fixture(name="lens_galaxy_prior")
def make_lens_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="source_galaxy_prior")
def make_source_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="model_mapper")
def make_model_mapper(test_config):
    return mm.ModelMapper(config=test_config)


@pytest.fixture(name="non_linear_optimizer")
def make_non_linear_optimizer():
    return MockNLO()


@pytest.fixture(name="model_stage")
def make_model_stage(lens_galaxy_prior, source_galaxy_prior, model_mapper, non_linear_optimizer):
    return pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior],
                            source_galaxy_priors=[source_galaxy_prior], pixelization=MockPixelization(),
                            non_linear_optimizer=non_linear_optimizer, model_mapper=model_mapper)


class TestModelStage:
    def test_setup(self, lens_galaxy_prior, source_galaxy_prior, model_mapper):
        pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior],
                         source_galaxy_priors=[source_galaxy_prior], pixelization=MockPixelization(),
                         non_linear_optimizer=MockNLO(), model_mapper=model_mapper)

        assert len(model_mapper.prior_models) == 2

    def test_run(self, model_stage, non_linear_optimizer):
        result = model_stage.run(MockImage(), MockMask())
        assert len(non_linear_optimizer.priors) == 2

        assert result.likelihood == 1
        assert result.lens_galaxies[0].redshift == 0.5
        assert result.source_galaxies[0].redshift == 0.5


class TestLinearPipeline:
    def test_simple_run(self):
        s1 = MockStage()
        s2 = MockStage()
        s3 = MockStage()

        pipeline = pl.LinearPipeline(s1, s2, s3)

        assert True not in map(lambda a: a.is_run, (s1, s2, s3))

        results = pipeline.run()

        assert len(results) == 3
        assert False not in map(lambda a: a.is_run, (s1, s2, s3))
