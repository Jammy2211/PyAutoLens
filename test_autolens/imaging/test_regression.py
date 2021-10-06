import autoarray as aa
import autofit as af


def test_pixelization_config():
    model = af.Model(aa.pix.Rectangular)
    assert model.shape[0].lower_limit == 32.1
    model.instance_from_prior_medians()
