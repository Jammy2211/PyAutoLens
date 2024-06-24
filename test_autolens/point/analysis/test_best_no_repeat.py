import pytest

from autolens.point.analysis import AnalysisBestNoRepeat


def test_best_match_analysis(grid, model):
    analysis = AnalysisBestNoRepeat(
        coordinates=[
            (0.0, 0.0),
        ],
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
        magnification_threshold=0.0,
    )

    assert analysis.log_likelihood_function(model.instance_from_prior_medians())


@pytest.mark.parametrize(
    "observed, predicted, likelihood",
    [
        ([(0.0, 0.0)], [(0.0, 0.0)], -2.076793740349318),
        ([(0.0, 0.0)], [(0.0, 0.1)], -2.326793740349318),
        ([(0.0, 0.0)], [(0.1, 0.1)], -2.576793740349318),
        ([(0.0, 0.0), (0.1, 0.1)], [(0.0, 0.0), (0.1, 0.1)], -2.076793740349318),
        ([(0.0, 0.0), (0.1, 0.1)], [(0.1, 0.1), (0.0, 0.0)], -2.076793740349318),
    ],
)
def test_likelihood__multiple_images(
    grid,
    model,
    observed,
    predicted,
    likelihood,
):
    analysis = AnalysisBestNoRepeat(
        coordinates=observed,
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
    )

    assert analysis._log_likelihood_for_coordinates(predicted) == likelihood
