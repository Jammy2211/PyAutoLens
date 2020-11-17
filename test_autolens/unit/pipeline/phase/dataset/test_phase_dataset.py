from os import path

import numpy as np
import pytest

import autolens as al
from autolens import exc
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__positions_are_input__are_used_in_analysis(
        self, image_7x7, noise_map_7x7, mask_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens dataset.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.2)
            ),
        )

        phase_imaging_7x7.modify_dataset(
            dataset=imaging_7x7, results=mock.MockResults()
        )
        phase_imaging_7x7.modify_settings(
            dataset=imaging_7x7, results=mock.MockResults()
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (
            analysis.masked_dataset.positions.in_list[0][0] == np.array([1.0, 1.0])
        ).all()
        assert (
            analysis.masked_dataset.positions.in_list[0][1] == np.array([2.0, 2.0])
        ).all()
        assert analysis.settings.settings_lens.positions_threshold == 0.2

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):
            imaging_7x7 = al.Imaging(
                image=image_7x7, noise_map=noise_map_7x7, positions=None
            )

            phase_imaging_7x7 = al.PhaseImaging(
                search=mock.MockSearch("test_phase"),
                settings=al.SettingsPhaseImaging(
                    settings_lens=al.SettingsLens(positions_threshold=0.2)
                ),
            )

            phase_imaging_7x7.modify_dataset(
                dataset=imaging_7x7, results=mock.MockResults()
            )
            phase_imaging_7x7.modify_settings(
                dataset=imaging_7x7, results=mock.MockResults()
            )

    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, imaging_7x7, mask_7x7
    ):
        # noinspection PyTypeChecker

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=100.0)
        )
        source_galaxy = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.Rectangular(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_inversion_class=al.Grid
                ),
                settings_pixelization=al.SettingsPixelization(use_border=True),
            ),
            search=mock.MockSearch("test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        analysis.masked_dataset.grid_inversion[4] = np.array([[500.0, 0.0]])

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.masked_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(97.19584, 1.0e-2)
        assert fit.inversion.mapper.grid[4][1] == pytest.approx(-3.699999, 1.0e-2)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_inversion_class=al.Grid
                ),
                settings_pixelization=al.SettingsPixelization(use_border=False),
            ),
            search=mock.MockSearch("test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        analysis.masked_dataset.grid_inversion[4] = np.array([300.0, 0.0])

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.masked_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(200.0, 1.0e-4)


class TestAutoPositions:
    def test__updates_correct_using_factor(
        self, imaging_7x7, image_7x7, noise_map_7x7, mask_7x7
    ):
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        # Auto positioning is OFF, so use input positions + threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.1)
            ),
        )

        results = mock.MockResults(max_log_likelihood_tracer=tracer)

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, but there are no previous results, so use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2, auto_positions_factor=2.0
                )
            ),
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). However, only one set of positions is computed from the previous
        # result, to use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2, auto_positions_factor=2.0
                )
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer,
            updated_positions=al.GridCoordinates(coordinates=[[(2.0, 2.0)]]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, but the tracer only has a single plane and thus no lensing, so use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2, auto_positions_factor=1.0
                )
            ),
        )

        tracer_x1_plane = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5)])

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer_x1_plane,
            updated_positions=al.GridCoordinates(
                coordinates=[[(2.0, 2.0), (3.0, 3.0)]]
            ),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). Multiple positions are available so these are now used.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2, auto_positions_factor=2.0
                )
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer,
            updated_positions=al.GridCoordinates(
                coordinates=[[(2.0, 2.0), (3.0, 3.0)]]
            ),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(2.0, 2.0), (3.0, 3.0)]]

        # Auto positioning is Off, but there are previous results with updated positions relative to the input
        # positions, so use those with their positions threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(2.0, 2.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.1)
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer,
            positions=al.GridCoordinates(coordinates=[[(3.0, 3.0), (4.0, 4.0)]]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [[(3.0, 3.0), (4.0, 4.0)]]

    def test__uses_auto_update_factor(self, image_7x7, noise_map_7x7, mask_7x7):
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        # Auto positioning is OFF, so use input positions + threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 1.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.1)
            ),
        )

        results = mock.MockResults(max_log_likelihood_tracer=tracer)

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)
        phase_imaging_7x7.modify_settings(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.settings.settings_lens.positions_threshold == 0.1

        # Auto positioning is ON, but there are no previous results, so use separate of postiions x positions factor..

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 0.0), (-1.0, 0.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.1, auto_positions_factor=1.0
                )
            ),
        )

        results = mock.MockResults(max_log_likelihood_tracer=tracer)

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)
        phase_imaging_7x7.modify_settings(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.settings.settings_lens.positions_threshold == 2.0

        # Auto position is ON, and same as above but with a factor of 3.0 which increases the threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 0.0), (-1.0, 0.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2, auto_positions_factor=3.0
                )
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer, updated_positions_threshold=0.2
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)
        phase_imaging_7x7.modify_settings(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.settings.settings_lens.positions_threshold == 6.0

        # Auto position is ON, and same as above but with a minimum auto positions threshold that rounds the value up.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.GridCoordinates([[(1.0, 0.0), (-1.0, 0.0)]]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(
                    positions_threshold=0.2,
                    auto_positions_factor=3.0,
                    auto_positions_minimum_threshold=10.0,
                )
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer, updated_positions_threshold=0.2
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)
        phase_imaging_7x7.modify_settings(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.settings.settings_lens.positions_threshold == 10.0

        # Auto positioning is ON, but positions are None and it cannot find new positions so no threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7, noise_map=noise_map_7x7, positions=None
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(auto_positions_factor=1.0)
            ),
        )

        results = mock.MockResults(max_log_likelihood_tracer=tracer)

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)
        phase_imaging_7x7.modify_settings(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.settings.settings_lens.positions_threshold == None
