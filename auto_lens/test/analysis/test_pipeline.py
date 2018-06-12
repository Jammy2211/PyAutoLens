from auto_lens.analysis import pipeline as pl
from auto_lens.analysis import galaxy_prior
from auto_lens.analysis import model_mapper as mm
import pytest
import os
import numpy as np


class MockResult:
    pass


class MockAnalysis:
    def __init__(self):
        self.is_run = False

    def run(self):
        self.is_run = True
        return MockResult()


class MockImage:
    pass


class MockPixelization:
    pass


class MockInstrumentation:
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


@pytest.fixture(name="model_analysis")
def make_model_analysis(lens_galaxy_prior, source_galaxy_prior, model_mapper, non_linear_optimizer):
    return pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior], source_galaxy_priors=[source_galaxy_prior],
                            non_linear_optimizer=non_linear_optimizer, model_mapper=model_mapper)


class TestModelAnalysis:
    def test_setup(self, lens_galaxy_prior, source_galaxy_prior, model_mapper):
        pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior],
                         source_galaxy_priors=[source_galaxy_prior],
                         non_linear_optimizer=MockNLO(), model_mapper=model_mapper)

        assert len(model_mapper.prior_models) == 2

    def test_run(self, model_analysis, non_linear_optimizer):
        result = model_analysis.run(MockImage(), MockMask(), MockPixelization(), MockInstrumentation())
        assert len(non_linear_optimizer.priors) == 2

        assert result.likelihood == 1
        assert result.lens_galaxies[0].redshift == 0.5
        assert result.source_galaxies[0].redshift == 0.5


class TestLinearPipeline:
    def test_simple_run(self):
        a1 = MockAnalysis()
        a2 = MockAnalysis()
        a3 = MockAnalysis()

        pipeline = pl.LinearPipeline(a1, a2, a3)

        assert True not in map(lambda a: a.is_run, (a1, a2, a3))

        results = pipeline.run()

        assert len(results) == 3
        assert False not in map(lambda a: a.is_run, (a1, a2, a3))


class MockHyperparameterAnalysis(object):
    pass


class TestMainPipeline:
    def test_main_pipeline(self, lens_galaxy_prior, source_galaxy_prior):
        pipeline = pl.MainPipeline(pl.ModelAnalysis([lens_galaxy_prior], [source_galaxy_prior]),
                                   hyperparameter_analysis=MockHyperparameterAnalysis())
        assert len(pipeline.run(MockImage(), MockMask(), MockPixelization(), MockInstrumentation())) == 2
