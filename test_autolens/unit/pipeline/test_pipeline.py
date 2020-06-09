import builtins

import autofit as af
import autolens as al
import numpy as np
import pytest
from autofit import Paths


class MockAnalysis:
    def __init__(self, number_galaxies, shape, value):
        self.number_galaxies = number_galaxies
        self.shape = shape
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.full(self.shape, self.value)]


class MockMask:
    pass


class Optimizer:
    def __init__(self, phase_name="dummy_phase"):
        self.phase_name = phase_name
        self.phase_path = ""


class DummyPhaseImaging(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, phase_tag=""):
        super().__init__(Paths(name=phase_name, tag=phase_tag))
        self.dataset = None
        self.positions = None
        self.results = None
        self.mask = None

        self.search = Optimizer(phase_name)

    def run(self, dataset, results, mask=None, positions=None, info=None):
        self.save_metadata(dataset)
        self.dataset = dataset
        self.results = results
        self.mask = mask
        self.positions = positions
        return af.Result(af.ModelInstance(), 1)


class MockImagingData(af.Dataset):
    def __init__(self, metadata=None):
        self._metadata = metadata or dict()

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def name(self) -> str:
        return "data_name"


class MockFile:
    def __init__(self):
        self.text = None
        self.filename = None

    def write(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(name="mock_files", autouse=True)
def make_mock_file(monkeypatch):
    files = []

    def mock_open(filename, flag, *args, **kwargs):
        assert flag in ("w+", "w+b", "a")
        file = MockFile()
        file.filename = filename
        files.append(file)
        return file

    monkeypatch.setattr(builtins, "open", mock_open)
    yield files


class TestMetaData:
    def test_files(self, mock_files):
        pipeline = al.PipelineDataset(
            "pipeline_name", DummyPhaseImaging(phase_name="phase_name")
        )
        pipeline.run(dataset=MockImagingData(), mask=MockMask())

        assert (
            mock_files[2].text
            == "phase=phase_name\nphase_tag=\npipeline=pipeline_name\npipeline_tag=\nnon_linear_search=search\ndataset_name=data_name"
        )

        assert "phase_name////non_linear.pickle" in mock_files[3].filename


class TestPassMask:
    def test_pass_mask(self):
        mask = MockMask()
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = al.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), mask=mask)

        assert phase_1.mask is mask
        assert phase_2.mask is mask


class TestPassPositions:
    def test_pass_positions(self):
        positions = [[(1.0, 1.0), (2.0, 2.0)]]
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = al.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), mask=MockMask(), positions=positions)

        assert phase_1.positions == positions
        assert phase_2.positions == positions


class TestPipelineImaging:
    def test_run_pipeline(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")

        pipeline = al.PipelineDataset("", phase_1, phase_2)

        pipeline.run(dataset=MockImagingData(), mask=MockMask())

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        phase_3 = DummyPhaseImaging("three")

        pipeline1 = al.PipelineDataset("", phase_1, phase_2)
        pipeline2 = al.PipelineDataset("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases


class DummyPhasePositions(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name):
        super().__init__(Paths(name=phase_name, tag=""))
        self.positions = None
        self.results = None
        self.pixel_scales = None
        self.search = Optimizer(phase_name)

    def run(self, positions, pixel_scales, results):
        self.save_metadata(MockImagingData())
        self.positions = positions
        self.pixel_scales = pixel_scales
        self.results = results
        return af.Result(af.ModelInstance(), 1)


class TestPipelinePositions:
    def test_run_pipeline(self):
        phase_1 = DummyPhasePositions(phase_name="one")
        phase_2 = DummyPhasePositions(phase_name="two")
        pipeline = al.PipelinePositions("", phase_1, phase_2)

        pipeline.run(positions=None, pixel_scales=None)

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhasePositions("one")
        phase_2 = DummyPhasePositions("two")
        phase_3 = DummyPhasePositions("three")

        pipeline1 = al.PipelinePositions("", phase_1, phase_2)
        pipeline2 = al.PipelinePositions("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
