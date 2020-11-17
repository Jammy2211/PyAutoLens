from os import path

import autofit as af
import autolens as al
from autolens import exc
import pytest
from astropy import cosmology as cosmo
from autolens.fit.fit import FitImaging
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestLogLikelihoodFunction:
    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 100.0), (200.0, 2.0)]])

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            ),
            settings=al.SettingsPhaseImaging(
                settings_lens=al.SettingsLens(positions_threshold=0.01)
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)


class TestFit:
    def test__fit_using_imaging(self, imaging_7x7, mask_7x7, samples_with_result):

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            search=mock.MockSearch(samples=samples_with_result),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        assert isinstance(result.instance.galaxies[0], al.Galaxy)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, imaging_7x7, mask_7x7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(lens=lens_galaxy),
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(sub_size=1)
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        masked_imaging = al.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7,
            settings=al.SettingsMaskedImaging(sub_size=1),
        )
        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, imaging_7x7, mask_7x7
    ):
        hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(lens=lens_galaxy),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings=al.SettingsPhaseImaging(
                settings_masked_imaging=al.SettingsMaskedImaging(sub_size=4)
            ),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        assert analysis.masked_dataset.mask.sub_size == 4

        masked_imaging = al.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7,
            settings=al.SettingsMaskedImaging(sub_size=4),
        )
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = FitImaging(
            masked_imaging=masked_imaging,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__uses_hyper_fit_correctly(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(intensity=1.0),
            mass=al.mp.SphericalIsothermal,
        )
        galaxies.source = al.Galaxy(redshift=1.0, light=al.lp.EllipticalSersic())

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        lens_hyper_image = al.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        hyper_model_image = al.Array.full(
            fill_value=0.5, shape_2d=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "lens"): lens_hyper_image}

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=al.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

        hyper_galaxy = al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        fit_likelihood = analysis.log_likelihood_function(instance=instance)

        g0 = al.Galaxy(
            redshift=0.5,
            light_profile=instance.galaxies.lens.light,
            mass_profile=instance.galaxies.lens.mass,
            hyper_galaxy=hyper_galaxy,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image=lens_hyper_image,
            hyper_minimum_value=0.0,
        )
        g1 = al.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        fit = FitImaging(masked_imaging=masked_imaging_7x7, tracer=tracer)

        assert (fit.tracer.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
        assert fit_likelihood == fit.log_likelihood

    def test__stochastic_histogram_for_instance(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.2)
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=5),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        lens_hyper_image = al.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        source_hyper_image = al.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        source_hyper_image[4] = 10.0
        hyper_model_image = al.Array.full(
            fill_value=0.5, shape_2d=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): lens_hyper_image,
            ("galaxies", "source"): source_hyper_image,
        }

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=al.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

        log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance, histogram_samples=2
        )

        assert log_evidences[0] != log_evidences[1]
