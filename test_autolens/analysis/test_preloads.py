import autofit as af
import autolens as al
from autolens import exc
from autolens.mock import mock
from autolens.analysis import preloads

import pytest


def test__preload_pixelization():

    result = mock.MockResult(max_log_likelihood_pixelization_grids_of_planes=1)

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(al.Galaxy), source=af.Model(al.Galaxy))
    )

    with pytest.raises(exc.PreloadException):
        preloads.preload_pixelization_grid_from(result=result, model=model)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy),
            source=af.Model(
                al.Galaxy,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        )
    )

    with pytest.raises(exc.PreloadException):
        preloads.preload_pixelization_grid_from(result=result, model=model)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy),
            source=af.Model(
                al.Galaxy,
                pixelization=al.pix.VoronoiBrightnessImage(),
                regularization=al.reg.AdaptiveBrightness(),
            ),
        )
    )

    preload_pixelization_grid = preloads.preload_pixelization_grid_from(
        result=result, model=model
    )

    assert preload_pixelization_grid == 1
