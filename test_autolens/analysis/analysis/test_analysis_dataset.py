from os import path
import os
import pytest

from autoconf import conf
from autoconf.dictable import from_json

import autofit as af
import autolens as al
from autolens import exc

directory = path.dirname(path.realpath(__file__))


def test__modify_before_fit__inversion_no_positions_likelihood__raises_exception(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(), regularization=al.reg.Constant()
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    with pytest.raises(exc.AnalysisException):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        positions_likelihood_list=[positions_likelihood],
        use_jax=False,
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)


def test__save_results__tracer_output_to_json(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5)
    source = al.Galaxy(redshift=1.0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    tracer = al.Tracer(galaxies=[lens, source])

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(max_log_likelihood_tracer=tracer, model=model),
    )

    tracer = from_json(file_path=paths._files_path / "tracer.json")

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[1].redshift == 1.0

    os.remove(paths._files_path / "tracer.json")
