from os import path

import numpy as np
import pytest

import autofit as af
from autofit.mapper.prior.prior import TuplePrior
import autolens as al
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


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
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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

        assert analysis.masked_dataset.positions.in_list == [(1.0, 1.0)]

        # Auto positioning is ON, but there are no previous results, so use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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

        assert analysis.masked_dataset.positions.in_list == [(1.0, 1.0)]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). However, only one set of positions is computed from the previous
        # result, to use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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
            updated_positions=al.Grid2DIrregular(grid=[(2.0, 2.0)]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [(1.0, 1.0)]

        # Auto positioning is ON, but the tracer only has a single plane and thus no lensing, so use input positions.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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
            updated_positions=al.Grid2DIrregular(grid=[(2.0, 2.0), (3.0, 3.0)]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [(1.0, 1.0)]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). Multiple positions are available so these are now used.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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
            updated_positions=al.Grid2DIrregular(grid=[(2.0, 2.0), (3.0, 3.0)]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [(2.0, 2.0), (3.0, 3.0)]

        # Auto positioning is Off, but there are previous results with updated positions relative to the input
        # positions, so use those with their positions threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(2.0, 2.0)]),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.1)
            ),
        )

        results = mock.MockResults(
            max_log_likelihood_tracer=tracer,
            positions=al.Grid2DIrregular(grid=[(3.0, 3.0), (4.0, 4.0)]),
            updated_positions_threshold=0.3,
        )

        phase_imaging_7x7.modify_dataset(dataset=imaging_7x7, results=results)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.positions.in_list == [(3.0, 3.0), (4.0, 4.0)]

    def test__uses_auto_update_factor(self, image_7x7, noise_map_7x7, mask_7x7):

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        # Auto positioning is OFF, so use input positions + threshold.

        imaging_7x7 = al.Imaging(
            image=image_7x7,
            noise_map=noise_map_7x7,
            positions=al.Grid2DIrregular([(1.0, 1.0)]),
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
            positions=al.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)]),
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
            positions=al.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)]),
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
            positions=al.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)]),
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


class TestExtensions:
    def test__extend_with_stochastic_phase__sets_up_model_correctly(self, mask_7x7):
        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5,
            light=al.lp.SphericalSersic(),
            mass=al.mp.SphericalIsothermal(),
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(),
            regularization=al.reg.AdaptiveBrightness(),
        )

        phase = al.PhaseImaging(
            search=mock.MockSearch(),
            galaxies=af.CollectionPriorModel(lens=al.GalaxyModel(redshift=0.5)),
        )

        phase_extended = phase.extend_with_stochastic_phase()

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)

        phase_extended = phase.extend_with_stochastic_phase(include_lens_light=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, af.UniformPrior)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)

        phase_extended = phase.extend_with_stochastic_phase(include_pixelization=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, af.UniformPrior)
        assert not isinstance(
            model.source.regularization.inner_coefficient, af.UniformPrior
        )

        phase_extended = phase.extend_with_stochastic_phase(include_regularization=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(
            model.source.regularization.inner_coefficient, af.UniformPrior
        )

        phase = al.PhaseInterferometer(
            search=mock.MockSearch(),
            real_space_mask=mask_7x7,
            galaxies=af.CollectionPriorModel(lens=al.GalaxyModel(redshift=0.5)),
        )

        phase_extended = phase.extend_with_stochastic_phase()

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)
