import numpy as np
import pytest

import autofit as af
import autolens as al

from autolens.imaging.model.result import ResultImaging


def test___linear_light_profiles_in_result(analysis_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.galaxy = al.Galaxy(redshift=0.5, bulge=al.lp_linear.Sersic())

    instance = af.ModelInstance()
    instance.tracer = al.Tracer(galaxies=galaxies)

    samples = al.m.MockSamples(max_log_likelihood_instance=instance)

    result = ResultImaging(samples=samples, analysis=analysis_imaging_7x7)

    assert not isinstance(
        result.max_log_likelihood_tracer.galaxies[0].bulge,
        al.lp_linear.LightProfileLinear,
    )
    assert result.max_log_likelihood_tracer.galaxies[
        0
    ].bulge.intensity == pytest.approx(0.002310735, 1.0e-4)
