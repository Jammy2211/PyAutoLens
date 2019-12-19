import builtins
import numpy as np
import pytest

import autofit as af
import autolens as al
from autofit import Paths
from autolens import exc


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
    def __init__(self, phase_name="dummy_phase"):
        self.phase_name = phase_name
        self.phase_path = ""


class DummyPhaseImaging(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, phase_tag=""):
        super().__init__(Paths(phase_name=phase_name, phase_tag=phase_tag))
        self.dataset = None
        self.positions = None
        self.results = None
        self.mask = None

        self.optimizer = Optimizer(phase_name)

    def run(self, dataset, results, mask=None, positions=None):
        self.dataset = dataset
        self.results = results
        self.mask = mask
        self.positions = positions
        self.assert_and_save_pickle()
        return af.Result(af.ModelInstance(), 1)


class MockImagingData(object):
    pass


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
        pipeline.run(MockImagingData(), data_name="data_name")

        assert (
            mock_files[1].text
            == "pipeline=pipeline_name\nphase=phase_name\ndataset_name=data_name\nphase_tag=\npipeline_tag=None"
        )

        assert "phase_name///optimizer.pickle" in mock_files[2].filename


class TestPassMask:
    def test_pass_mask(self):
        mask = MockMask()
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = al.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), mask=mask)

        assert phase_1.mask is mask
        assert phase_2.mask is mask


class TestPassPositions(object):
    def test_pass_positions(self):
        positions = [[(1.0, 1.0), (2.0, 2.0)]]
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = al.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), positions=positions)

        assert phase_1.positions == positions
        assert phase_2.positions == positions


class TestPipelineImaging(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")

        pipeline = al.PipelineDataset("", phase_1, phase_2)

        pipeline.run(MockImagingData())

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        phase_3 = DummyPhaseImaging("three")

        pipeline1 = al.PipelineDataset("", phase_1, phase_2)
        pipeline2 = al.PipelineDataset("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases

    def test__hyper_mode_on__must_receive_mask(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")

        pipeline = al.PipelineDataset("", phase_1, phase_2, hyper_mode=False)

        pipeline.run(MockImagingData())

        pipeline = al.PipelineDataset("", phase_1, phase_2, hyper_mode=True)

        with pytest.raises(exc.PhaseException):
            pipeline.run(MockImagingData())

        pipeline.run(dataset=MockImagingData, mask=1.0)


class DummyPhasePositions(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name):
        super().__init__(Paths(phase_name=phase_name, phase_tag=""))
        self.positions = None
        self.results = None
        self.pixel_scales = None
        self.optimizer = Optimizer(phase_name)

    def run(self, positions, pixel_scales, results):
        self.positions = positions
        self.pixel_scales = pixel_scales
        self.results = results
        return af.Result(af.ModelInstance(), 1)


class TestPipelinePositions(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhasePositions(phase_name="one")
        phase_2 = DummyPhasePositions(phase_name="two")
        pipeline = al.PipelinePositions("", phase_1, phase_2)

        pipeline.run(None, None)

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhasePositions("one")
        phase_2 = DummyPhasePositions("two")
        phase_3 = DummyPhasePositions("three")

        pipeline1 = al.PipelinePositions("", phase_1, phase_2)
        pipeline2 = al.PipelinePositions("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
