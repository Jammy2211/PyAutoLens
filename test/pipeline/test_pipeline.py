from src.pipeline import pipeline as pl
from src.pipeline import phase as ph
from src.autopipe import model_mapper
from src.autopipe import non_linear


class DummyPhase(object):
    def __init__(self):
        self.masked_image = None
        self.last_result = None

    def run(self, masked_image, last_result=None):
        self.masked_image = masked_image
        self.last_result = last_result
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        pipeline = pl.Pipeline(phase_1, phase_2)

        pipeline.run(None)

        assert phase_1.last_result is None
        assert phase_2.last_result is not None
