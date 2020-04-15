from os import path

import pytest

import autofit as af
import autolens as al
from test_autolens.mock import mock_pipeline

directory = path.dirname(path.realpath(__file__))


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
