import numpy as np
from autolens.point.fit.positions.image.pair_repeat import Fit


def test_andrew_implementation():
    data = np.array([(0.0, 0.0), (1.0, 0.0)])
    model_positions = np.array([(-1.0749, -1.1), (1.19117, 1.175)])

    error = 1.0

    assert (
        Fit(
            data,
            model_positions,
            np.array(
                [error, error],
            ),
        ).log_likelihood()
        == -4.40375330990644
    )
