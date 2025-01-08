import numpy as np
import pytest

from autolens.point.fit.positions.image.pair_repeat import Fit


@pytest.fixture
def data():
    return np.array([(0.0, 0.0), (1.0, 0.0)])


@pytest.fixture
def noise_map():
    return np.array([1.0, 1.0])


def test_andrew_implementation(
    data,
    noise_map,
):
    model_positions = np.array([(-1.0749, -1.1), (1.19117, 1.175)])

    assert (
        Fit(
            data=data,
            noise_map=noise_map,
            model_positions=model_positions,
        ).log_likelihood()
        == -4.40375330990644
    )


def test_nan_model_positions(
    data,
    noise_map,
):
    model_positions = np.array([(-1.0749, -1.1), (1.19117, 1.175), (np.nan, np.nan)])

    assert (
        Fit(
            data=data,
            noise_map=noise_map,
            model_positions=model_positions,
        ).log_likelihood()
        == -4.40375330990644
    )
