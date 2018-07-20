from src.pipeline import phase as ph
import pytest


class MockPhase(ph.Phase):
    def __init__(self, optimizer):
        super(MockPhase, self).__init__(optimizer)
        self.mock_class = None


class MockClass(object):
    def __init__(self, one):
        self.one = one


class MockOptimizer(object):
    def __init__(self):
        self.variable = object()
        self.constant = object()


@pytest.fixture(name="phase")
def make_phase():
    return MockPhase(optimizer=MockOptimizer())


@pytest.fixture(name="mock_instance")
def make_mock_instance():
    return MockClass(1)


class TestPhase(object):
    def test_set_constants(self, phase, mock_instance):
        phase.mock_class = mock_instance
        assert phase.optimizer.constant.mock_instance is not None
