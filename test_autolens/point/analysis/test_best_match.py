from autolens.point.analysis import AnalysisBestMatch


def test_best_match_analysis(grid, model):
    analysis = AnalysisBestMatch(
        coordinates=[
            (0.0, 0.0),
        ],
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
        magnification_threshold=0.0,
    )

    assert analysis.log_likelihood_function(model.instance_from_prior_medians())
