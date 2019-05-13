from autolens.pipeline import phase as ph
from autofit.mapper import model_mapper as mm
from autofit.optimize import non_linear as nl


class TestCase(object):
    def test_lens_source_phase_constant(self):
        phase = ph.LensSourcePlanePhase("test")

        assert isinstance(phase.constant, mm.ModelInstance)

    def test_non_linear(self):
        optimizer = nl.NonLinearOptimizer("test")

        assert isinstance(optimizer.constant, mm.ModelInstance)

    def test_multinest(self):
        optimizer = nl.MultiNest("test")

        assert isinstance(optimizer.constant, mm.ModelInstance)

    def test_abstract_phase(self):
        phase = ph.AbstractPhase("test")

        assert isinstance(phase.constant, mm.ModelInstance)
