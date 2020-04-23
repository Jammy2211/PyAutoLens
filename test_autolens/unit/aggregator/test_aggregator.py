from os import path
import numpy as np
import pytest

import autofit as af
import autolens as al
from test_autolens.mock import mock_pipeline

import os

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return "{}/files/".format(os.path.dirname(os.path.realpath(__file__)))


def test__masked_imaging_generator_from_aggregator(imaging_7x7, mask_7x7):

    phase_imaging_7x7 = al.PhaseImaging(
        non_linear_class=mock_pipeline.MockNLO,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        phase_name="test_phase_aggregator",
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.phase_output_path)

    masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()


def test__tracer_generator_from_aggregator(imaging_7x7, mask_7x7):

    phase_imaging_7x7 = al.PhaseImaging(
        non_linear_class=mock_pipeline.MockNLO,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        phase_name="test_phase_aggregator",
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.phase_output_path)

    tracer_gen = al.agg.Tracer(aggregator=agg)

    for tracer in tracer_gen:

        assert tracer.galaxies[0].redshift == 0.5
        assert tracer.galaxies[0].light.centre == (0.0, 1.0)
        assert tracer.galaxies[1].redshift == 1.0


def test__fit_imaging_generator_from_aggregator(imaging_7x7, mask_7x7):

    phase_imaging_7x7 = al.PhaseImaging(
        non_linear_class=mock_pipeline.MockNLO,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        phase_name="test_phase_aggregator",
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.phase_output_path)

    fit_imaging_gen = al.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.masked_imaging.imaging.image == imaging_7x7.image).all()


def test__results_array_from_results_file(path):

    array = al.agg.results_array_from_grid_phase_results(file_results=f"{path}results")

    assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert array.pixel_scale == 2.0
    assert list(array.extent_of_zoomed_array(buffer=0)) == [-2.0, 2.0, -2.0, 2.0]
