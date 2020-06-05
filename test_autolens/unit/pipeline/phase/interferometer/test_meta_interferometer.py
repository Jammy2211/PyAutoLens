from os import path

import autolens as al
import pytest
from test_autogalaxy.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


def test__masked_imaging__settings_inputs_are_used_in_masked_imaging(
    interferometer_7, mask_7x7
):

    interferometer_7.positions = al.GridCoordinates([[[1.0, 1.0]]])

    phase_interferometer_7 = al.PhaseInterferometer(
        phase_name="phase_interferometer_7",
        real_space_mask=mask_7x7,
        settings=al.PhaseSettingsInterferometer(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=3,
            signal_to_noise_limit=1.0,
            bin_up_factor=2,
            inversion_pixel_limit=100,
            primary_beam_shape_2d=(3, 3),
            positions_threshold=0.3,
            inversion_uses_border=False,
            inversion_stochastic=True,
        ),
    )

    assert phase_interferometer_7.meta_dataset.settings.sub_size == 3
    assert phase_interferometer_7.meta_dataset.settings.signal_to_noise_limit == 1.0
    assert phase_interferometer_7.meta_dataset.settings.bin_up_factor == 2
    assert phase_interferometer_7.meta_dataset.settings.inversion_pixel_limit == 100
    assert phase_interferometer_7.meta_dataset.settings.primary_beam_shape_2d == (3, 3)
    assert phase_interferometer_7.meta_dataset.settings.positions_threshold == 0.3
    assert phase_interferometer_7.meta_dataset.settings.inversion_uses_border == False
    assert phase_interferometer_7.meta_dataset.settings.inversion_stochastic == True

    analysis = phase_interferometer_7.make_analysis(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, al.Grid)
    assert isinstance(analysis.masked_dataset.grid_inversion, al.Grid)
    assert isinstance(analysis.masked_dataset.transformer, al.TransformerNUFFT)
    assert analysis.masked_dataset.positions_threshold == 0.3
    assert analysis.masked_dataset.inversion_uses_border == False
    assert analysis.masked_dataset.inversion_stochastic == True

    phase_interferometer_7 = al.PhaseInterferometer(
        phase_name="phase_interferometer_7",
        real_space_mask=mask_7x7,
        settings=al.PhaseSettingsInterferometer(
            grid_class=al.GridIterate,
            sub_size=3,
            fractional_accuracy=0.99,
            sub_steps=[2],
            transformer_class=al.TransformerDFT,
        ),
    )

    analysis = phase_interferometer_7.make_analysis(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, al.GridIterate)
    assert analysis.masked_dataset.grid.sub_size == 1
    assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
    assert analysis.masked_dataset.grid.sub_steps == [2]
    assert isinstance(analysis.masked_dataset.transformer, al.TransformerDFT)
