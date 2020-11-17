from os import path
import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__masked_interferometer__settings_inputs_are_used_in_masked_interferometer(
        self, interferometer_7, mask_7x7
    ):
        phase_interferometer_7 = al.PhaseInterferometer(
            search=mock.MockSearch("phase_interferometer_7"),
            settings=al.SettingsPhaseInterferometer(
                settings_masked_interferometer=al.SettingsMaskedInterferometer(
                    grid_class=al.Grid,
                    grid_inversion_class=al.Grid,
                    sub_size=3,
                    signal_to_noise_limit=1.0,
                ),
                settings_pixelization=al.SettingsPixelization(
                    use_border=False, is_stochastic=True
                ),
                settings_inversion=al.SettingsInversion(use_linear_operators=True),
            ),
            real_space_mask=mask_7x7,
        )

        assert (
            phase_interferometer_7.settings.settings_masked_interferometer.sub_size == 3
        )
        assert (
            phase_interferometer_7.settings.settings_masked_interferometer.signal_to_noise_limit
            == 1.0
        )
        assert phase_interferometer_7.settings.settings_pixelization.use_border == False
        assert (
            phase_interferometer_7.settings.settings_pixelization.is_stochastic == True
        )
        assert (
            phase_interferometer_7.settings.settings_inversion.use_linear_operators
            == True
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, al.Grid)
        assert isinstance(analysis.masked_dataset.grid_inversion, al.Grid)
        assert isinstance(analysis.masked_dataset.transformer, al.TransformerNUFFT)

        phase_interferometer_7 = al.PhaseInterferometer(
            settings=al.SettingsPhaseInterferometer(
                settings_masked_interferometer=al.SettingsMaskedInterferometer(
                    grid_class=al.GridIterate,
                    sub_size=3,
                    fractional_accuracy=0.99,
                    sub_steps=[2],
                    transformer_class=al.TransformerDFT,
                )
            ),
            search=mock.MockSearch("phase_interferometer_7"),
            real_space_mask=mask_7x7,
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, al.GridIterate)
        assert analysis.masked_dataset.grid.sub_size == 1
        assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
        assert analysis.masked_dataset.grid.sub_steps == [2]
        assert isinstance(analysis.masked_dataset.transformer, al.TransformerDFT)

    def test__masks_visibilities_and_noise_map_correctly(
        self, phase_interferometer_7, interferometer_7, visibilities_mask_7x2
    ):
        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )

        assert (
            analysis.masked_interferometer.visibilities == interferometer_7.visibilities
        ).all()
        assert (
            analysis.masked_interferometer.noise_map == interferometer_7.noise_map
        ).all()

    def test__phase_info_is_made(
        self, phase_interferometer_7, interferometer_7, visibilities_mask_7x2
    ):
        phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )

        file_phase_info = path.join(
            phase_interferometer_7.search.paths.output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        search = phase_info.readline()
        sub_size = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert search == "Optimizer = MockSearch \n"
        assert sub_size == "Sub-grid size = 2 \n"
        assert positions_threshold == "Positions Threshold = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )

    def test__phase_can_receive_hyper_image_and_noise_maps(self, mask_7x7):
        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=al.Redshift),
                lens1=al.GalaxyModel(redshift=al.Redshift),
            ),
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
            search=mock.MockSearch("test_phase"),
            real_space_mask=mask_7x7,
        )

        instance = phase_interferometer_7.model.instance_from_vector([0.1, 0.2, 0.3])

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_background_noise.noise_scale == 0.3


class TestHyperMethods:
    def test__phase_is_extended_with_hyper_phases__sets_up_hyper_images(
        self, interferometer_7, mask_7x7
    ):
        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(redshift=0.5)
        galaxies.source = al.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): al.Array.ones(shape_2d=(3, 3), pixel_scales=1.0),
            ("galaxies", "source"): al.Array.full(
                fill_value=2.0, shape_2d=(3, 3), pixel_scales=1.0
            ),
        }

        hyper_galaxy_visibilities_path_dict = {
            ("galaxies", "lens"): al.Visibilities.full(fill_value=4.0, shape_1d=(7,)),
            ("galaxies", "source"): al.Visibilities.full(fill_value=5.0, shape_1d=(7,)),
        }

        results = mock.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array.full(
                fill_value=3.0, shape_2d=(3, 3), pixel_scales=1.0
            ),
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=al.Visibilities.full(
                fill_value=6.0, shape_1d=(7,)
            ),
            mask=mask_7x7,
            use_as_hyper_dataset=True,
        )

        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, hyper_galaxy=al.HyperGalaxy)
            ),
            search=mock.MockSearch("test_phase"),
            real_space_mask=mask_7x7,
        )

        phase_interferometer_7.extend_with_multiple_hyper_phases(
            setup_hyper=al.SetupHyper()
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=mask_7x7, results=results
        )

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "lens")].in_2d
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].in_2d
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.in_2d == 3.0 * np.ones((3, 3))).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "lens")]
            == 4.0 * np.ones((7, 2))
        ).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "source")]
            == 5.0 * np.ones((7, 2))
        ).all()

        assert (analysis.hyper_model_visibilities == 6.0 * np.ones((7, 2))).all()
