import os

import autofit as af
import autolens as al
import numpy as np
import pytest
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = os.path.dirname(os.path.realpath(__file__))


class TestMakeAnalysis:
    def test__masks_visibilities_and_noise_map_correctly(
        self, phase_interferometer_7, interferometer_7, visibilities_mask_7x2
    ):
        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock_pipeline.MockResults(),
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
            results=mock_pipeline.MockResults(),
        )

        file_phase_info = "{}/{}".format(
            phase_interferometer_7.optimizer.paths.output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        optimizer = phase_info.readline()
        sub_size = phase_info.readline()
        primary_beam_shape_2d = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert optimizer == "Optimizer = MockNLO \n"
        assert sub_size == "Sub-grid size = 2 \n"
        assert primary_beam_shape_2d == "Primary Beam shape = None \n"
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
            real_space_mask=mask_7x7,
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
            non_linear_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_interferometer_7.model.instance_from_vector([0.1, 0.2, 0.3])

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_background_noise.noise_scale == 0.3

    def test__log_likelihood_cap(self, interferometer_7, mask_7x7):

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseInterferometer(
            galaxies=[lens_galaxy],
            real_space_mask=mask_7x7,
            settings=al.PhaseSettingsInterferometer(
                grid_class=al.Grid, sub_size=1, log_likelihood_cap=100.0
            ),
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert analysis.log_likelihood_cap == 100.0


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

        results = mock_pipeline.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array.full(fill_value=3.0, shape_2d=(3, 3)),
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
            real_space_mask=mask_7x7,
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        phase_interferometer_7.extend_with_multiple_hyper_phases()

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
