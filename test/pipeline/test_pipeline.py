from autolens.pipeline import pipeline as pl
from pipelines import profile_pipeline
from autolens.pipeline import phase as ph
from autolens.autopipe import model_mapper
from autolens.autopipe import non_linear
from autolens.imaging import image as im
from autolens.profiles import light_profiles
from autolens.profiles import mass_profiles
from autolens.analysis import galaxy_prior as gp
from autolens.analysis import galaxy as g
import numpy as np
import pytest


class MockAnalysis(object):

    def __init__(self, number_galaxies, shape, value):
        self.number_galaxies = number_galaxies
        self.shape = shape
        self.value = value

    def galaxy_images_for_model(self, model):
        return self.number_galaxies*[np.full(self.shape, self.value)]


class DummyPhase(object):
    def __init__(self):
        self.masked_image = None
        self.previous_results = None
        self.phase_name = "dummy_phase"

    def run(self, masked_image, previous_results):
        self.masked_image = masked_image
        self.previous_results = previous_results
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        pipeline = pl.PipelineImaging("", phase_1, phase_2)

        pipeline.run(None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        phase_3 = DummyPhase()

        pipeline1 = pl.PipelineImaging("", phase_1, phase_2)
        pipeline2 = pl.PipelineImaging("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases