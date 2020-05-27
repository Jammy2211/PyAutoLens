import os
from os import path

import autofit as af
import autolens as al
import numpy as np
import pytest
from test_autolens.mock import mock_pipeline

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return "{}/files/".format(os.path.dirname(os.path.realpath(__file__)))


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

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    tracer_gen = al.agg.Tracer(aggregator=agg)

    for tracer in tracer_gen:

        assert tracer.galaxies[0].redshift == 0.5
        assert tracer.galaxies[0].light.centre == (0.0, 1.0)
        assert tracer.galaxies[1].redshift == 1.0


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

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()


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

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    fit_imaging_gen = al.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.masked_imaging.imaging.image == imaging_7x7.image).all()


def test__masked_interferometer_generator_from_aggregator(interferometer_7, mask_7x7):

    phase_interferometer_7x7 = al.PhaseInterferometer(
        non_linear_class=mock_pipeline.MockNLO,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        real_space_mask=mask_7x7,
        phase_name="test_phase_aggregator",
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    masked_interferometer_gen = al.agg.MaskedInterferometer(aggregator=agg)

    for masked_interferometer in masked_interferometer_gen:
        assert (
            masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer.real_space_mask == mask_7x7).all()


def test__fit_interferometer_generator_from_aggregator(interferometer_7, mask_7x7):

    phase_interferometer_7x7 = al.PhaseInterferometer(
        non_linear_class=mock_pipeline.MockNLO,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        phase_name="test_phase_aggregator",
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    fit_interferometer_gen = al.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (
            fit_interferometer.masked_interferometer.real_space_mask == mask_7x7
        ).all()
