from os import path

import autolens as al
from autolens import exc
from autolens.mock import mock
import pytest
from astropy import cosmology as cosmo
from autolens.fit.fit import FitInterferometer

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestLogLikelihoodFunction:
    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_interferometer_7, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        interferometer_7.positions = al.GridCoordinates([[(1.0, 100.0), (200.0, 2.0)]])

        phase_interferometer_7 = al.PhaseInterferometer(
            real_space_mask=mask_7x7,
            galaxies=dict(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            ),
            settings=al.SettingsPhaseInterferometer(
                settings_lens=al.SettingsLens(positions_threshold=0.01)
            ),
            search=mock.MockSearch("test_phase"),
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)


class TestFit:
    def test__fit_using_interferometer(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2, samples_with_result
    ):
        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            search=mock.MockSearch("test_phase", samples=samples_with_result),
            real_space_mask=mask_7x7,
        )

        result = phase_interferometer_7.run(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )
        assert isinstance(result.instance.galaxies[0], al.Galaxy)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(lens=lens_galaxy),
            cosmology=cosmo.FLRW,
            settings=al.SettingsPhaseInterferometer(
                settings_masked_interferometer=al.SettingsMaskedInterferometer(
                    sub_size=2
                )
            ),
            search=mock.MockSearch("test_phase"),
            real_space_mask=mask_7x7,
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=mask_7x7,
            settings=al.SettingsMaskedInterferometer(sub_size=2),
        )
        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitInterferometer(
            masked_interferometer=masked_interferometer, tracer=tracer
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_interferometer_7 = al.PhaseInterferometer(
            galaxies=dict(lens=lens_galaxy),
            hyper_background_noise=hyper_background_noise,
            settings=al.SettingsPhaseInterferometer(
                settings_masked_interferometer=al.SettingsMaskedInterferometer(
                    sub_size=4
                )
            ),
            search=mock.MockSearch("test_phase"),
            real_space_mask=mask_7x7,
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        assert analysis.masked_interferometer.real_space_mask.sub_size == 4

        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=mask_7x7,
            settings=al.SettingsMaskedInterferometer(sub_size=4),
        )
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = FitInterferometer(
            masked_interferometer=masked_interferometer,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit
