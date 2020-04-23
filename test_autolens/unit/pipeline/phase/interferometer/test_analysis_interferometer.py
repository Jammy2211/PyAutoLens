from os import path

import pytest
from astropy import cosmology as cosmo

import autolens as al
from autolens.fit.fit import FitInterferometer
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestFit:
    def test__fit_using_interferometer(
        self, interferometer_7, mask_7x7, visibilities_mask_7x2
    ):
        phase_interferometer_7 = al.PhaseInterferometer(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            real_space_mask=mask_7x7,
            phase_name="test_phase_test_fit",
        )

        result = phase_interferometer_7.run(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock_pipeline.MockResults(),
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
            real_space_mask=mask_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            sub_size=2,
            phase_name="test_phase",
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock_pipeline.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        real_space_mask = phase_interferometer_7.meta_dataset.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=real_space_mask,
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
            real_space_mask=mask_7x7,
            galaxies=[lens_galaxy],
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            sub_size=4,
            phase_name="test_phase",
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7x2,
            results=mock_pipeline.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        real_space_mask = phase_interferometer_7.meta_dataset.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        assert real_space_mask.sub_size == 4

        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=real_space_mask,
        )
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = FitInterferometer(
            masked_interferometer=masked_interferometer,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit
