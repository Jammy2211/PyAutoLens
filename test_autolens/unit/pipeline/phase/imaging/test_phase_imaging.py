from os import path

import numpy as np
import pytest

import autofit as af
import autolens as al
from autofit.mapper.prior.prior import TuplePrior
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_and_pixelizations(self):
        phase = al.PhaseImaging(search=mock.MockSearch())

        phase_extended = phase.extend_with_stochastic_phase()

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5,
            light=al.lp.SphericalSersic(),
            mass=al.mp.SphericalIsothermal(),
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage,
            regularization=al.reg.AdaptiveBrightness,
        )

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.einstein_radius, af.UniformPrior)
        assert isinstance(model.lens.light.intensity, float)


class TestMakeAnalysis:
    def test__masked_imaging__settings_inputs_are_used_in_masked_imaging(
        self, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_class=al.Grid,
                    grid_inversion_class=al.Grid,
                    sub_size=3,
                    signal_to_noise_limit=1.0,
                    bin_up_factor=2,
                    psf_shape_2d=(3, 3),
                ),
                settings_pixelization=al.SettingsPixelization(
                    use_border=False, is_stochastic=True
                ),
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.settings.settings_masked_imaging.sub_size == 3
        assert (
            phase_imaging_7x7.settings.settings_masked_imaging.signal_to_noise_limit
            == 1.0
        )
        assert phase_imaging_7x7.settings.settings_masked_imaging.bin_up_factor == 2
        assert phase_imaging_7x7.settings.settings_masked_imaging.psf_shape_2d == (3, 3)
        assert phase_imaging_7x7.settings.settings_pixelization.use_border == False
        assert phase_imaging_7x7.settings.settings_pixelization.is_stochastic == True

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, al.Grid)
        assert isinstance(analysis.masked_dataset.grid_inversion, al.Grid)

        phase_imaging_7x7 = al.PhaseImaging(
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_class=al.GridIterate,
                    sub_size=3,
                    fractional_accuracy=0.99,
                    sub_steps=[2],
                )
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, al.GridIterate)
        assert analysis.masked_dataset.grid.sub_size == 1
        assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
        assert analysis.masked_dataset.grid.sub_steps == [2]

    def test__masked_imaging__signal_to_noise_limit(self, imaging_7x7, mask_7x7_1_pix):
        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from(
            signal_to_noise_limit=1.0
        )

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch(),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    signal_to_noise_limit=1.0
                )
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock.MockResults()
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
        binned_up_imaging = imaging_7x7.binned_up_from(bin_up_factor=2)

        binned_up_mask = mask_7x7_1_pix.binned_mask_from_bin_up_factor(bin_up_factor=2)

        phase_imaging_7x7 = al.PhaseImaging(
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(bin_up_factor=2)
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock.MockResults()
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (
            analysis.masked_dataset.psf == (1.0 / 9.0) * binned_up_imaging.psf
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_dataset.mask == binned_up_mask).all()

    def test__grid_classes_input__used_in_masked_imaging(self, imaging_7x7, mask_7x7):
        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch(),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_inversion_class=al.Grid
                )
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        assert isinstance(analysis.masked_imaging.grid, al.Grid)
        assert isinstance(analysis.masked_imaging.grid_inversion, al.Grid)

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch(),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_class=al.GridIterate,
                    grid_inversion_class=al.GridIterate,
                    fractional_accuracy=0.2,
                    sub_steps=[2, 3],
                )
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        assert isinstance(analysis.masked_imaging.grid, al.GridIterate)
        assert analysis.masked_imaging.grid.fractional_accuracy == 0.2
        assert analysis.masked_imaging.grid.sub_steps == [2, 3]
        assert isinstance(analysis.masked_imaging.grid_inversion, al.GridIterate)
        assert analysis.masked_imaging.grid_inversion.fractional_accuracy == 0.2
        assert analysis.masked_imaging.grid_inversion.sub_steps == [2, 3]

        phase_imaging_7x7 = al.PhaseImaging(
            search=mock.MockSearch(),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(
                    grid_class=al.GridInterpolate,
                    grid_inversion_class=al.GridInterpolate,
                    pixel_scales_interp=0.1,
                )
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        assert isinstance(analysis.masked_imaging.grid, al.GridInterpolate)
        assert analysis.masked_imaging.grid.pixel_scales_interp == (0.1, 0.1)
        assert isinstance(analysis.masked_imaging.grid_inversion, al.GridInterpolate)
        assert analysis.masked_imaging.grid_inversion.pixel_scales_interp == (0.1, 0.1)

    def test__masks_image_and_noise_map_correctly(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_7x7.image.in_2d * np.invert(mask_7x7)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_7x7.noise_map.in_2d * np.invert(mask_7x7)
        ).all()

    def test___phase_info_is_made(self, phase_imaging_7x7, imaging_7x7, mask_7x7):
        phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        file_phase_info = path.join(phase_imaging_7x7.search.paths.output_path, "phase.info")

        phase_info = open(file_phase_info, "r")

        search = phase_info.readline()
        sub_size = phase_info.readline()
        psf_shape_2d = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert search == "Optimizer = MockSearch \n"
        assert sub_size == "Sub-grid size = 2 \n"
        assert psf_shape_2d == "PSF shape = None \n"
        assert positions_threshold == "Positions Threshold = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )


class TestExtensions:
    def test__phase_can_receive_hyper_image_and_noise_maps(self):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=al.Redshift),
                lens1=al.GalaxyModel(redshift=al.Redshift),
            ),
            hyper_image_sky=al.hyper_data.HyperImageSky,
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
            search=mock.MockSearch(),
        )

        instance = phase_imaging_7x7.model.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.sky_scale == 0.3
        assert instance.hyper_background_noise.noise_scale == 0.4

    def test__extend_with_hyper_phases__sets_up_hyper_dataset_from_results(
        self, imaging_7x7, mask_7x7
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

        results = mock.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array.full(
                fill_value=3.0, shape_2d=(3, 3), pixel_scales=1.0
            ),
            mask=mask_7x7,
            use_as_hyper_dataset=True,
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, hyper_galaxy=al.HyperGalaxy)
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
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

    def test__extend_with_stochastic_phase__sets_up_model_correctly(self):
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

        phase = al.PhaseImaging(search=mock.MockSearch())

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
