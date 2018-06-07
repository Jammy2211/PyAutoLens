from auto_lens.analysis import pipeline
from auto_lens.analysis import galaxy_prior
import pytest


class MockNLO:
    def __init__(self):
        self.priors = None
        self.fitness_function = None

    def run(self, fitness_function, priors):
        self.fitness_function = fitness_function
        self.priors = priors


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


class MockModelMapper:
    def __init__(self):
        self.classes = {}
        self.priors_ordered_by_id = [MockPrior(), MockPrior()]

    def add_class(self, name, cls):
        self.classes[name] = cls
        return MockPriorModel(name, cls)


class MockModelInstance:
    pass


@pytest.fixture(name="lens_galaxy_prior")
def make_lens_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="source_galaxy_prior")
def make_source_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="model_mapper")
def make_model_mapper():
    return MockModelMapper()


@pytest.fixture(name="non_linear_optimizer")
def make_non_linear_optimizer():
    return MockNLO()


@pytest.fixture(name="model_analysis")
def make_model_analysis(lens_galaxy_prior, source_galaxy_prior, model_mapper, non_linear_optimizer):
    return pipeline.ModelAnalysis(image=MockImage(), lens_galaxy_priors=[lens_galaxy_prior],
                                  source_galaxy_priors=[source_galaxy_prior], pixelization=MockPixelization(),
                                  model_mapper=model_mapper, non_linear_optimizer=non_linear_optimizer)


class TestModelAnalysis:
    def test_setup(self, lens_galaxy_prior, source_galaxy_prior, model_mapper):
        pipeline.ModelAnalysis(image=MockImage(), lens_galaxy_priors=[lens_galaxy_prior],
                               source_galaxy_priors=[source_galaxy_prior], pixelization=MockPixelization(),
                               model_mapper=model_mapper, non_linear_optimizer=MockNLO())

        assert len(model_mapper.classes) == 2

    def test_run(self, model_analysis, non_linear_optimizer):
        model_analysis.run()
        assert len(non_linear_optimizer.priors) == 2
