import pytest

from autolens.point.analysis import AnalysisClosestPointSource


def test_log_likelihood(grid, model):
    analysis = AnalysisClosestPointSource(
        coordinates=[
            (0.0, 0.0),
        ],
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
    )

    assert analysis.log_likelihood_function(model.instance_from_prior_medians())


@pytest.mark.parametrize(
    "observed, predicted, likelihood",
    [
        ([(0.0, 0.0)], [(0.0, 0.0)], 0.0),
        ([(0.0, 0.0)], [(0.0, 0.1)], -0.5),
        ([(0.0, 0.0)], [(0.1, 0.1)], -1.0),
        ([(0.0, 0.0), (0.1, 0.1)], [(0.0, 0.0), (0.1, 0.1)], 0.0),
        ([(0.0, 0.0), (0.1, 0.1)], [(0.1, 0.1), (0.0, 0.0)], 0.0),
    ],
)
def test_likelihood__multiple_images(
    grid,
    model,
    observed,
    predicted,
    likelihood,
):
    analysis = AnalysisClosestPointSource(
        coordinates=observed,
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
    )

    assert analysis._log_likelihood_for_coordinates(predicted) == pytest.approx(
        likelihood,
        abs=0.01,
    )
