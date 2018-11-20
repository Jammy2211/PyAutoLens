import numpy as np

from autofit.core import model_mapper
from autofit.core import non_linear
from autolens.pipeline import pipeline as pl


class MockAnalysis(object):

    def __init__(self, number_galaxies, shape, value):
        self.number_galaxies = number_galaxies
        self.shape = shape
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.full(self.shape, self.value)]


class MockMask(object):
    pass


class DummyPhaseImaging(object):
    def __init__(self):
        self.masked_image = None
        self.positions = None
        self.previous_results = None
        self.phase_name = "dummy_phase"
        self.mask = None

    def run(self, masked_image, previous_results, mask=None):
        print(mask)
        self.masked_image = masked_image
        self.previous_results = previous_results
        self.mask = mask
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPassMask(object):
    def test_pass_mask(self):
        mask = MockMask()
        phase_1 = DummyPhaseImaging()
        phase_2 = DummyPhaseImaging()
        pipeline = pl.PipelineImaging("", phase_1, phase_2)
        pipeline.run(None, mask)

        assert phase_1.mask is mask
        assert phase_2.mask is mask


class TestPipelineImaging(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhaseImaging()
        phase_2 = DummyPhaseImaging()
        pipeline = pl.PipelineImaging("", phase_1, phase_2)

        pipeline.run(None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhaseImaging()
        phase_2 = DummyPhaseImaging()
        phase_3 = DummyPhaseImaging()

        pipeline1 = pl.PipelineImaging("", phase_1, phase_2)
        pipeline2 = pl.PipelineImaging("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases


class DummyPhasePositions(object):
    def __init__(self):
        self.positions = None
        self.previous_results = None
        self.pixel_scale = None
        self.phase_name = "dummy_phase"

    def run(self, positions, pixel_scale, previous_results):
        self.positions = positions
        self.pixel_scale = pixel_scale
        self.previous_results = previous_results
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipelinePositions(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhasePositions()
        phase_2 = DummyPhasePositions()
        pipeline = pl.PipelinePositions("", phase_1, phase_2)

        pipeline.run(None, None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhasePositions()
        phase_2 = DummyPhasePositions()
        phase_3 = DummyPhasePositions()

        pipeline1 = pl.PipelinePositions("", phase_1, phase_2)
        pipeline2 = pl.PipelinePositions("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
