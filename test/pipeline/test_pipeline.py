from src.pipeline import pipeline as pl
from src.autopipe import model_mapper
from src.autopipe import non_linear


class DummyPhase(object):
    def __init__(self):
        self.masked_image = None
        self.previous_results = None
        self.name = "dummy_phase"

    def run(self, masked_image, previous_results):
        self.masked_image = masked_image
        self.previous_results = previous_results
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        pipeline = pl.Pipeline(phase_1, phase_2)

        pipeline.run(None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        phase_3 = DummyPhase()

        pipeline1 = pl.Pipeline(phase_1, phase_2)
        pipeline2 = pl.Pipeline(phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
