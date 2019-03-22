import builtins
import numpy as np
import pytest

from autofit.mapper import model_mapper
from autofit.optimize import non_linear
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


class Optimizer(object):
    def __init__(self):
        self.phase_name = "dummy_phase"


class DummyPhaseImaging(object):
    def __init__(self, phase_name, phase_path=None):
        self.data = None
        self.positions = None
        self.results = None
        self.phase_name = phase_name
        self.phase_path = phase_path or phase_name
        self.mask = None

        self.optimizer = Optimizer()

    def run(self, data, results, mask=None, positions=None):
        self.data = data
        self.results = results
        self.mask = mask
        self.positions = positions
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class MockCCDData(object):
    def __init__(self, name):
        self.name = name


class MockFile(object):
    def __init__(self):
        self.text = None
        self.filename = None

    def write(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(name="mock_file", autouse=True)
def make_mock_file(monkeypatch):
    file = MockFile()

    def mock_open(filename, flag):
        assert flag == "w+"
        file.filename = filename
        return file

    monkeypatch.setattr(builtins, 'open', mock_open)
    return file


class TestMetaData(object):
    def test_name(self, mock_file):
        pipeline = pl.PipelineImaging("pipeline_name", DummyPhaseImaging("phase_name", "phase_path"))
        pipeline.run(MockCCDData("data_name"))

        assert "phase_name/.metadata" in mock_file.filename
        assert mock_file.text == "pipeline=pipeline_name\nphase=phase_name\nlens=data_name"


class TestPassMask(object):
    def test_pass_mask(self):
        mask = MockMask()
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = pl.PipelineImaging("", phase_1, phase_2)
        pipeline.run(data=MockCCDData(""), mask=mask)

        assert phase_1.mask is mask
        assert phase_2.mask is mask


class TestPassPositions(object):
    def test_pass_positions(self):
        positions = [[[1.0, 1.0], [2.0, 2.0]]]
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = pl.PipelineImaging("", phase_1, phase_2)
        pipeline.run(data=MockCCDData(""), positions=positions)

        assert phase_1.positions == positions
        assert phase_2.positions == positions


class TestPipelineImaging(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = pl.PipelineImaging("", phase_1, phase_2)

        pipeline.run(MockCCDData(""))

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        phase_3 = DummyPhaseImaging("three")

        pipeline1 = pl.PipelineImaging("", phase_1, phase_2)
        pipeline2 = pl.PipelineImaging("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases


class DummyPhasePositions(object):
    def __init__(self, phase_name):
        self.positions = None
        self.results = None
        self.pixel_scale = None
        self.phase_name = phase_name
        self.optimizer = Optimizer()

    def run(self, positions, pixel_scale, results):
        self.positions = positions
        self.pixel_scale = pixel_scale
        self.results = results
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipelinePositions(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhasePositions("one")
        phase_2 = DummyPhasePositions("two")
        pipeline = pl.PipelinePositions("", phase_1, phase_2)

        pipeline.run(None, None)

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhasePositions("one")
        phase_2 = DummyPhasePositions("two")
        phase_3 = DummyPhasePositions("three")

        pipeline1 = pl.PipelinePositions("", phase_1, phase_2)
        pipeline2 = pl.PipelinePositions("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
