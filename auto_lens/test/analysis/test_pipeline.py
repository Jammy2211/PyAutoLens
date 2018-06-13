from auto_lens.analysis import pipeline as pl
from auto_lens.analysis import galaxy_prior
from auto_lens.analysis import model_mapper as mm
from auto_lens import instrumentation as inst
import pytest
import os
import numpy as np


class MockResult:
    def __init__(self, image=None, mask=None, pixelization=None, instrumentation=None, lens_galaxies=None,
                 source_galaxies=None):
        self.image = image
        self.mask = mask
        self.pixelization = pixelization
        self.instrumentation = instrumentation
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies


class MockPixelization:
    def __init__(self, number_clusters, regularization_coefficient=1):
        self.number_clusters = number_clusters
        self.regularization_coefficient = regularization_coefficient


class MockModelAnalysis:
    def __init__(self):
        self.image = None
        self.mask = None
        self.pixelization = None
        self.instrumentation = None

    def run(self, image, mask, pixelization, instrumentation):
        self.image = image
        self.mask = mask
        self.pixelization = pixelization
        self.instrumentation = instrumentation
        return MockResult(lens_galaxies=[MockGalaxy()], source_galaxies=[MockGalaxy(), MockGalaxy()])


class MockHyperparameterAnalysis(object):
    def __init__(self):
        self.image = None
        self.mask = None
        self.lens_galaxies = None
        self.source_galaxies = None

    def run(self, image, mask, lens_galaxies, source_galaxies):
        self.image = image
        self.mask = mask
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies
        return MockResult(pixelization=MockPixelization(0), instrumentation=inst.Instrumentation(0))


class MockImage:
    pass


class MockPrior:
    pass


class MockGalaxy:
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def deflections_at_coordinates(self, coordinates):
        return 1


class MockPriorModel:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.centre = "centre for {}".format(name)
        self.phi = "phi for {}".format(name)


class MockModelInstance:
    pass


class MockNLO:
    def __init__(self, arr):
        self.priors = None
        self.fitness_function = None
        self.arr = arr

    def run(self, fitness_function, priors):
        self.fitness_function = fitness_function
        self.priors = priors
        fitness_function(self.arr)


class MockMask:
    # noinspection PyMethodMayBeStatic
    def compute_grid_coords_image(self):
        return np.array([[-1., -1.], [1., 1.]])


@pytest.fixture(name='test_config')
def make_test_config():
    path = "{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/config")
    print(path)
    return mm.Config(config_folder_path=path)


@pytest.fixture(name="lens_galaxy_prior")
def make_lens_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="source_galaxy_prior")
def make_source_galaxy_prior():
    return galaxy_prior.GalaxyPrior()


@pytest.fixture(name="model_mapper")
def make_model_mapper(test_config):
    return mm.ModelMapper(config=test_config)


@pytest.fixture(name="model_analysis")
def make_model_analysis(lens_galaxy_prior, source_galaxy_prior, model_mapper):
    return pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior], source_galaxy_priors=[source_galaxy_prior],
                            non_linear_optimizer=MockNLO([0.5, 0.5]), model_mapper=model_mapper)


class TestModelAnalysis:
    def test_setup(self, lens_galaxy_prior, source_galaxy_prior, model_mapper):
        pl.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior],
                         source_galaxy_priors=[source_galaxy_prior],
                         non_linear_optimizer=MockNLO([0.5, 0.5]), model_mapper=model_mapper)

        assert len(model_mapper.prior_models) == 2

    def test_run(self, model_analysis):
        result = model_analysis.run(MockImage(), MockMask(), MockPixelization(0), inst.Instrumentation(0))
        assert len(model_analysis.non_linear_optimizer.priors) == 2

        assert result.likelihood == 1
        assert result.lens_galaxies[0].redshift == 0.5
        assert result.source_galaxies[0].redshift == 0.5


class TestHyperparameterAnalysis:
    def test_setup(self, model_mapper):
        pl.HyperparameterAnalysis(MockPixelization, inst.Instrumentation, model_mapper, MockNLO([0.5, 0.5, 0.5]))

        assert len(model_mapper.prior_models) == 2

    def test_run(self, model_mapper):
        hyperparameter_analysis = pl.HyperparameterAnalysis(MockPixelization, inst.Instrumentation, model_mapper,
                                                            MockNLO([0.5, 0.5, 0.5]))

        result = hyperparameter_analysis.run(MockImage(), MockMask(), [MockGalaxy()], [MockGalaxy()])
        assert len(hyperparameter_analysis.non_linear_optimizer.priors) == 3

        assert result.likelihood == 1
        assert result.pixelization.number_clusters == 0.5
        assert result.instrumentation.param == 0.5


class TestMainPipeline:
    def test_main_pipeline(self):
        hyperparameter_analysis = MockHyperparameterAnalysis()
        model_analysis = MockModelAnalysis()
        # noinspection PyTypeChecker
        pipeline = pl.MainPipeline(model_analysis, hyperparameter_analysis=hyperparameter_analysis)
        results = pipeline.run(MockImage(), MockMask(), MockPixelization(0), inst.Instrumentation(0))
        assert len(results) == 2
        assert len(hyperparameter_analysis.source_galaxies) == 2
        assert len(hyperparameter_analysis.lens_galaxies) == 1

        assert len(results[0]) == 1
        assert len(results[1]) == 1

        assert len(results[0][0].source_galaxies) == 2
        assert len(results[0][0].lens_galaxies) == 1


class TestAnalysis:
    def test_setup(self):
        analysis = pl.Analysis(lens_galaxy_priors=[galaxy_prior.GalaxyPrior()],
                               non_linear_optimizer=MockNLO([0.5]), model_mapper=mm.ModelMapper())

        analysis.run(image=MockImage(), source_galaxies=[MockGalaxy()], mask=MockMask(),
                     pixelization=MockPixelization(0), instrumentation=inst.Instrumentation(0))

        with pytest.raises(AssertionError):
            analysis.run(image=MockImage(), mask=MockMask(), pixelization=MockPixelization(0),
                         instrumentation=inst.Instrumentation(0))

        with pytest.raises(AssertionError):
            analysis.run(image=MockImage(), source_galaxies=[MockGalaxy()], lens_galaxies=[MockGalaxy()],
                         mask=MockMask(), pixelization=MockPixelization(0), instrumentation=inst.Instrumentation(0))
