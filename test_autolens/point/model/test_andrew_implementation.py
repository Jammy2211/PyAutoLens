try:
    import jax

    JAX_INSTALLED = True
except ImportError:
    JAX_INSTALLED = False

import numpy as np
import pytest

from autolens.point.fit.positions.image.pair_repeat import Fit


@pytest.fixture
def data():
    return np.array([(0.0, 0.0), (1.0, 0.0)])


@pytest.fixture
def noise_map():
    return np.array([1.0, 1.0])


@pytest.fixture
def fit(data, noise_map):
    model_positions = np.array(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
        ]
    )

    return Fit(
        data=data,
        noise_map=noise_map,
        model_positions=model_positions,
    )


def test_andrew_implementation(fit):
    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.log_likelihood() == -4.40375330990644


@pytest.mark.skipif(not JAX_INSTALLED, reason="JAX is not installed")
def test_jax(fit):
    assert jax.jit(fit.log_likelihood)() == -4.40375330990644


def test_nan_model_positions(
    data,
    noise_map,
):
    model_positions = np.array(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (np.nan, np.nan),
        ]
    )
    fit = Fit(
        data=data,
        noise_map=noise_map,
        model_positions=model_positions,
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.log_likelihood() == -4.40375330990644


def test_duplicate_model_position(
    data,
    noise_map,
):
    model_positions = np.array(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (1.19117, 1.175),
        ]
    )
    fit = Fit(
        data=data,
        noise_map=noise_map,
        model_positions=model_positions,
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [-1.14237812, -0.87193683],
    )
    assert fit.log_likelihood() == -4.211539531047171
