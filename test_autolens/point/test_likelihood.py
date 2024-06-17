from autolens.point.analysis import AllToAllPointSourceAnalysis
import autolens as al
import autofit as af


def test_likelihood():
    analysis = AllToAllPointSourceAnalysis(
        [
            (0.0, 0.0),
        ],
        0.1,
        al.Grid2D.uniform(
            shape_native=(100, 100),
            pixel_scales=0.05,
        ),
        pixel_scale_precision=0.025,
    )

    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
    source = af.Model(al.ps.PointSourceChi)

    model = af.Collection(lens=lens, source=source)
    assert analysis.log_likelihood_function(model.instance_from_prior_medians())
