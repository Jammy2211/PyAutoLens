import autolens as al
import numpy as np
import pytest


class MockTracerPositions:
    def __init__(self, positions, noise=None):
        self.positions = positions
        self.noise = noise


class TestPositionFit:
    def test__x1_positions__mock_position_tracer__maximum_separation_is_correct(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(2)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 3.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(1.0) + np.square(3.0))

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [-9.0, -4.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_positions__mock_position_tracer__maximum_separation_is_correct(
        self
    ):
        tracer = MockTracerPositions(
            positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]])]
        )
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(
            positions=[np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])]
        )
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(
            positions=[np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])]
        )
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(
            positions=[
                np.array(
                    [
                        [-2.0, -4.0],
                        [1.0, 3.0],
                        [0.1, 0.1],
                        [-0.1, -0.1],
                        [0.3, 0.4],
                        [-0.6, 0.5],
                    ]
                )
            ]
        )
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(
            positions=[np.array([[8.0, 4.0], [8.0, 4.0], [-9.0, -4.0]])]
        )
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_sets_of_positions__multiple_sets_of_max_distances(self):
        tracer = MockTracerPositions(
            positions=[
                np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]),
            ]
        )

        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)

        assert fit.maximum_separations[0] == 1.0
        assert fit.maximum_separations[1] == np.sqrt(18)
        assert fit.maximum_separations[2] == np.sqrt(18)

    def test__likelihood__is_sum_of_separations_divided_by_noise(self):
        tracer = MockTracerPositions(
            positions=[
                np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]),
            ]
        )

        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.chi_squared_map[0] == 1.0
        assert fit.chi_squared_map[1] == pytest.approx(18.0, 1e-4)
        assert fit.chi_squared_map[2] == pytest.approx(18.0, 1e-4)
        assert fit.figure_of_merit == pytest.approx(-0.5 * (1.0 + 18 + 18), 1e-4)

        fit = al.LensPositionFit(positions=tracer.positions, noise_map=2.0)
        assert fit.chi_squared_map[0] == (1.0 / 2.0) ** 2.0
        assert fit.chi_squared_map[1] == pytest.approx(
            (np.sqrt(18.0) / 2.0) ** 2.0, 1e-4
        )
        assert fit.chi_squared_map[2] == pytest.approx(
            (np.sqrt(18.0) / 2.0) ** 2.0, 1e-4
        )
        assert fit.figure_of_merit == pytest.approx(
            -0.5
            * (
                (1.0 / 2.0) ** 2.0
                + (np.sqrt(18.0) / 2.0) ** 2.0
                + (np.sqrt(18.0) / 2.0) ** 2.0
            ),
            1e-4,
        )

    def test__threshold__if_not_met_returns_ray_tracing_exception(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = al.LensPositionFit(positions=tracer.positions, noise_map=1.0)

        assert fit.maximum_separation_within_threshold(threshold=100.0)
        assert not fit.maximum_separation_within_threshold(threshold=0.1)
