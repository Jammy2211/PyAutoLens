import pytest

from autolens.point.analysis import AnalysisAllToAllPointSource
import autolens as al
import autofit as af


@pytest.fixture
def grid():
    return al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,
    )


@pytest.fixture
def model():
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        mass=al.mp.Isothermal,
    )
    source = af.Model(al.ps.PointSourceChi)

    return af.Collection(
        lens=lens,
        source=source,
    )


def test_likelihood(grid, model):
    analysis = AnalysisAllToAllPointSource(
        coordinates=[
            (0.0, 0.0),
        ],
        error=0.1,
        grid=grid,
        pixel_scale_precision=0.025,
    )

    assert analysis.log_likelihood_function(model.instance_from_prior_medians())
