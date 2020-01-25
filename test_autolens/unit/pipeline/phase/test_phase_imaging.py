import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
import autolens as al
from autolens.fit.fit import ImagingFit
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
        "{}/../test_files/config/phase_imaging_7x7".format(directory)
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
    def test__make_analysis__masks_image_and_noise_map_correctly(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)

        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_7x7.image.in_2d * np.invert(mask_7x7)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_7x7.noise_map.in_2d * np.invert(mask_7x7)
        ).all()

    def test__make_analysis__phase_info_is_made(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)

        file_phase_info = "{}/{}".format(
            phase_imaging_7x7.optimizer.paths.phase_output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        optimizer = phase_info.readline()
        sub_size = phase_info.readline()
        psf_shape_2d = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert optimizer == "Optimizer = MockNLO \n"
        assert sub_size == "Sub-grid size = 2 \n"
        assert psf_shape_2d == "PSF shape = None \n"
        assert positions_threshold == "Positions Threshold = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )

    def test__fit_using_imaging(self, imaging_7x7, mask_7x7):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            phase_name="test_phase_test_fit",
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)

    def test__modify_image(self, imaging_7x7, mask_7x7):
        class MyPhase(al.PhaseImaging):
            def modify_image(self, image, results):
                assert imaging_7x7.image.shape_2d == image.shape_2d
                image = al.array.full(
                    fill_value=20.0, shape_2d=(7, 7), pixel_scales=1.0
                )
                return image

        phase_imaging_7x7 = MyPhase(phase_name="phase_imaging_7x7")

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)
        assert (
            analysis.masked_dataset.image.in_2d
            == 20.0 * np.ones(shape=(7, 7)) * np.invert(mask_7x7)
        ).all()
        assert (analysis.masked_dataset.image.in_1d == 20.0 * np.ones(shape=9)).all()

    def test__masked_imaging_signal_to_noise_limit(self, imaging_7x7, mask_7x7_1_pix):
        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", signal_to_noise_limit=1.0
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.1
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", signal_to_noise_limit=0.1
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

    def test__masked_imaging_is_binned_up(self, imaging_7x7, mask_7x7_1_pix):
        binned_up_imaging = imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2)

        binned_up_mask = mask_7x7_1_pix.mapping.binned_mask_from_bin_up_factor(
            bin_up_factor=2
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", bin_up_factor=2
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()
        assert (analysis.masked_dataset.psf == binned_up_imaging.psf).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_dataset.mask == binned_up_mask).all()

        masked_imaging = al.masked.imaging(imaging=imaging_7x7, mask=mask_7x7_1_pix)

        binned_up_masked_imaging = masked_imaging.binned_from_bin_up_factor(
            bin_up_factor=2
        )

        assert (
            analysis.masked_dataset.image.in_2d
            == binned_up_masked_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()
        assert (analysis.masked_dataset.psf == binned_up_masked_imaging.psf).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == binned_up_masked_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_dataset.mask == binned_up_masked_imaging.mask).all()

        assert (
            analysis.masked_dataset.image.in_1d == binned_up_masked_imaging.image.in_1d
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_1d
            == binned_up_masked_imaging.noise_map.in_1d
        ).all()

    def test__phase_can_receive_hyper_image_and_noise_maps(self):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=al.Redshift),
                lens1=al.GalaxyModel(redshift=al.Redshift),
            ),
            hyper_image_sky=al.hyper_data.HyperImageSky,
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_imaging_7x7.model.instance_from_physical_vector(
            [0.1, 0.2, 0.3, 0.4]
        )

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.sky_scale == 0.3
        assert instance.hyper_background_noise.noise_scale == 0.4

    def test__extended_with_hyper_and_pixelizations(self):

        phase_no_pixelization = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO, phase_name="test_phase"
        )

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_no_pixelization

        # This phase does not have a pixelization, so even though inversion=True it will not be extended

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert phase_extended == phase_no_pixelization

        phase_with_pixelization = al.PhaseImaging(
            galaxies=dict(
                source=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                )
            ),
            optimizer_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase
        assert type(phase_extended.hyper_phases[1]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True, hyper_galaxy_phase_first=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == al.InversionPhase

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, imaging_7x7, mask_7x7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            sub_size=1,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        masked_imaging = al.masked.imaging(imaging=imaging_7x7, mask=mask_7x7)
        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, imaging_7x7, mask_7x7
    ):
        hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy],
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            sub_size=4,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_imaging_7x7.meta_imaging_fit.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        assert mask.sub_size == 4

        masked_imaging = al.masked.imaging(imaging=imaging_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = ImagingFit(
            masked_imaging=masked_imaging,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.likelihood == fit_figure_of_merit
