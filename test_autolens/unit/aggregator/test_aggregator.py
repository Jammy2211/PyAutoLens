import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
import autolens as al
from autolens.fit.fit import FitImaging
from test_autolens.mock import mock_pipeline

directory = path.dirname(path.realpath(__file__))


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.dataset_path = directory


def test__dataset_generator_from_aggregator(imaging_7x7, mask_7x7):
    clean_images()

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

    dataset_gen = al.agg.Dataset(aggregator=agg)

    for dataset in dataset_gen:
        assert (dataset.image == imaging_7x7.image).all()


def test__mask_generator_from_aggregator(imaging_7x7, mask_7x7):
    clean_images()

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

    mask_gen = al.agg.Mask(aggregator=agg)

    for mask in mask_gen:
        assert (mask == mask_7x7).all()


def test__masked_imaging_generator_from_aggregator(imaging_7x7, mask_7x7):
    clean_images()

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

    clean_images()

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

    clean_images()

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
        assert fit_imaging.likelihood == pytest.approx(-1517.01, 1.0e-2)
