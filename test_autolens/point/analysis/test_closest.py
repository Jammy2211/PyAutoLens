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
