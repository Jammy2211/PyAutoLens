from auto_lens.analysis import analysis as a
from auto_lens.analysis import galaxy_prior
from auto_lens.analysis import model_mapper as mm
import pytest
import os
import numpy as np


# noinspection PyMissingConstructor
class MockResult(a.Analysis.Result):
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


class MockInstrumentation(object):
    def __init__(self, param=0):
        self.param = param


class MockModelAnalysis:
    def __init__(self):
        self.image = None
        self.mask = None
        self.pixelization = None
        self.instrumentation = None
        self.missing_attributes = ['pixelization, instrumentation']

    def run(self, image, mask, pixelization=None, instrumentation=None):
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
        self.missing_attributes = ['lens_galaxies', 'source_galaxies']

    def run(self, image, mask, lens_galaxies=None, source_galaxies=None):
        self.image = image
        self.mask = mask
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies
        return MockResult(pixelization=MockPixelization(0), instrumentation=MockInstrumentation(0))


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
    def __init__(self, num):
        self.fitness_function = None
        self.arr = [0.5 for _ in range(num)]

    def run(self, fitness_function):
        self.fitness_function = fitness_function
        fitness_function(self.arr)


class MockMask:
    # noinspection PyMethodMayBeStatic
    def compute_grid_coords_image(self):
        return np.array([[-1., -1.], [1., 1.]])


@pytest.fixture(name='test_config')
def make_test_config():
    path = "{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/config")
    return mm.Config(config_folder_path=path)


@pytest.fixture(name="model_mapper")
def make_model_mapper(test_config):
    return mm.ModelMapper(config=test_config)


@pytest.fixture(name="model_analysis")
def make_model_analysis(model_mapper):
    lens_galaxy_prior = galaxy_prior.GalaxyPrior("lens_galaxy", model_mapper)
    source_galaxy_prior = galaxy_prior.GalaxyPrior("source_galaxy", model_mapper)
    return a.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior], source_galaxy_priors=[source_galaxy_prior],
                           non_linear_optimizer=MockNLO(2), model_mapper=model_mapper)


# class TestModelAnalysis:
#     def test_setup(self, model_mapper):
#         lens_galaxy_prior = galaxy_prior.GalaxyPrior("lens_galaxy", model_mapper)
#         source_galaxy_prior = galaxy_prior.GalaxyPrior("source_galaxy", model_mapper)
#         a.ModelAnalysis(lens_galaxy_priors=[lens_galaxy_prior],
#                         source_galaxy_priors=[source_galaxy_prior],
#                         non_linear_optimizer=MockNLO(2), model_mapper=model_mapper)
#
#         assert len(model_mapper.prior_models) == 2
#
#     def test_run(self, model_analysis):
#         result = model_analysis.run(MockImage(), MockMask(), pixelization=MockPixelization(0),
#                                     instrumentation=MockInstrumentation(0))
#         # assert len(model_analysis.non_linear_optimizer.priors) == 2
#
#         assert result.likelihood == 1
#         assert result.lens_galaxies[0].redshift == 0.5
#         assert result.source_galaxies[0].redshift == 0.5
#
#
# class TestHyperparameterAnalysis:
#     def test_setup(self, model_mapper):
#         a.HyperparameterAnalysis(MockPixelization, MockInstrumentation, model_mapper, MockNLO(3))
#
#         assert len(model_mapper.prior_models) == 2
#
#     def test_run(self, model_mapper):
#         hyperparameter_analysis = a.HyperparameterAnalysis(MockPixelization, MockInstrumentation, model_mapper,
#                                                            MockNLO(3))
#
#         result = hyperparameter_analysis.run(MockImage(), MockMask(), lens_galaxies=[MockGalaxy()],
#                                              source_galaxies=[MockGalaxy()])
#         # assert len(hyperparameter_analysis.non_linear_optimizer.priors) == 3
#
#         assert result.likelihood == 1
#         assert result.pixelization.number_clusters == 0.5
#         assert result.instrumentation.param == 0.5
#
#
# class TestAnalysis:
#     def test_setup(self):
#         model_mapper = mm.ModelMapper()
#         analysis = a.Analysis(model_mapper, lens_galaxy_priors=[galaxy_prior.GalaxyPrior("lens_galaxy", model_mapper)],
#                               non_linear_optimizer=MockNLO(1))
#
#         analysis.run(image=MockImage(), source_galaxies=[MockGalaxy()], mask=MockMask(),
#                      pixelization=MockPixelization(0), instrumentation=MockInstrumentation(0))
#
#     def test_run(self):
#         model_mapper = mm.ModelMapper()
#         analysis = a.Analysis(model_mapper, lens_galaxy_priors=[galaxy_prior.GalaxyPrior("lens_galaxy", model_mapper)],
#                               non_linear_optimizer=MockNLO(1))
#
#         result = analysis.run(image=MockImage(), source_galaxies=[MockGalaxy()], mask=MockMask(),
#                               pixelization=MockPixelization(0), instrumentation=MockInstrumentation(0))
#
#         assert result.pixelization is not None
