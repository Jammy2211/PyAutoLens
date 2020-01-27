import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
import autolens as al
from autolens.fit.fit import InterferometerFit
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/config/phase_interferometer_7".format(directory)
    )


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.dataset_path = directory


class TestPhase(object):
    def test__make_analysis__masks_visibilities_and_noise_map_correctly(
        self, phase_interferometer_7, interferometer_7, visibilities_mask_7x2
    ):
        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )

        assert (
            analysis.masked_interferometer.visibilities == interferometer_7.visibilities
        ).all()
        assert (
            analysis.masked_interferometer.noise_map == interferometer_7.noise_map
        ).all()

    def test__make_analysis__phase_info_is_made(
        self, phase_interferometer_7, interferometer_7, visibilities_mask_7x2
    ):
        phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )

        file_phase_info = "{}/{}".format(
            phase_interferometer_7.optimizer.paths.phase_output_path, "phase.info"
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

    def test__fit_using_interferometer(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):

        phase_interferometer_7 = al.PhaseInterferometer(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            real_space_mask=mask_7x7,
            phase_name="test_phase_test_fit",
        )

        result = phase_interferometer_7.run(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )
        assert isinstance(result.instance.galaxies[0], al.Galaxy)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)

    def test_modify_visibilities(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        class MyPhase(al.PhaseInterferometer):
            def modify_visibilities(self, visibilities, results):
                assert interferometer_7.visibilities.shape_1d == visibilities.shape_1d
                visibilities = al.visibilities.full(fill_value=20.0, shape_1d=(7,))
                return visibilities

        phase_interferometer_7 = MyPhase(
            phase_name="phase_interferometer_7", real_space_mask=mask_7x7
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )
        assert (
            analysis.masked_dataset.visibilities == 20.0 * np.ones(shape=(7, 2))
        ).all()

    def test__phase_can_receive_hyper_image_and_noise_maps(self, mask_7x7):
        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=al.Redshift),
                lens1=al.GalaxyModel(redshift=al.Redshift),
            ),
            real_space_mask=mask_7x7,
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_interferometer_7.model.instance_from_physical_vector(
            [0.1, 0.2, 0.3]
        )

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_background_noise.noise_scale == 0.3

    def test__extended_with_hyper_and_pixelizations(self, phase_interferometer_7):
        phase_extended = phase_interferometer_7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_interferometer_7

        phase_extended = phase_interferometer_7.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_interferometer_7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase

        phase_extended = phase_interferometer_7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_interferometer_7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == al.InversionPhase

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_interferometer_7 = al.PhaseInterferometer(
            real_space_mask=mask_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            sub_size=2,
            phase_name="test_phase",
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        real_space_mask = phase_interferometer_7.meta_interferometer_fit.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        masked_interferometer = al.masked.interferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=real_space_mask,
        )
        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.fit(masked_dataset=masked_interferometer, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_interferometer_7 = al.PhaseInterferometer(
            real_space_mask=mask_7x7,
            galaxies=[lens_galaxy],
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            sub_size=4,
            phase_name="test_phase",
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=visibilities_mask_7x2
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        real_space_mask = phase_interferometer_7.meta_interferometer_fit.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        assert real_space_mask.sub_size == 4

        masked_interferometer = al.masked.interferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=real_space_mask,
        )
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = InterferometerFit(
            masked_interferometer=masked_interferometer,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.likelihood == fit_figure_of_merit
