from os import path

import autolens as al
import numpy as np
import pytest
from test_autolens import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__results_of_phase_include_mask__available_as_property(
        self, imaging_7x7, mask_7x7, samples_with_result
    ):

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)),
            search=mock.MockSearch(samples=samples_with_result),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(sub_size=2)
            ),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (result.mask == mask_7x7).all()

    def test__results_of_phase_include_positions__available_as_property(
        self, imaging_7x7, mask_7x7, samples_with_result
    ):

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase", search=mock.MockSearch(samples=samples_with_result)
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.positions == None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)),
            search=mock.MockSearch(samples=samples_with_result),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=1.0)
            ),
        )

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 1.0)]])

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (result.positions[0] == np.array([1.0, 1.0])).all()

    def test__results_of_phase_include_pixelization__available_as_property(
        self, imaging_7x7, mask_7x7
    ):
        lens = al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))
        source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiMagnification(shape=(2, 3)),
            regularization=al.reg.Constant(),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            settings=al.SettingsPhaseImaging(),
            search=mock.MockSearch(samples=samples),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(result.pixelization_prior_model, al.pix.VoronoiMagnification)
        assert result.pixelization_prior_model.shape == (2, 3)

        lens = al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))
        source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=6),
            regularization=al.reg.Constant(),
        )

        source.hyper_galaxy_image = np.ones(9)

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            settings=al.SettingsPhaseImaging(),
            search=mock.MockSearch(samples=samples),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(result.pixelization_prior_model, al.pix.VoronoiBrightnessImage)
        assert result.pixelization_prior_model.pixels == 6

    def test__results_of_phase_include_pixelization_grid__available_as_property(
        self, imaging_7x7, mask_7x7
    ):
        galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase_2",
            galaxies=dict(lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)),
            search=mock.MockSearch(samples=samples),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.max_log_likelihood_pixelization_grids_of_planes == [None]

        lens = al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))
        source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=6),
            regularization=al.reg.Constant(),
        )

        source.hyper_galaxy_image = np.ones(9)

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase_2",
            galaxies=dict(lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)),
            settings=al.SettingsPhaseImaging(),
            search=mock.MockSearch(samples=samples),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.max_log_likelihood_pixelization_grids_of_planes[-1].shape == (
            6,
            2,
        )
