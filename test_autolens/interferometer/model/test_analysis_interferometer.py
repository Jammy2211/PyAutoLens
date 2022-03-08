from os import path
import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens import exc

from autolens.interferometer.model.result import ResultInterferometer

directory = path.dirname(path.realpath(__file__))


class TestAnalysisInterferometer:
    def test__make_result__result_interferometer_is_returned(self, interferometer_7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

        instance = model.instance_from_prior_medians()

        samples = al.m.MockSamples(max_log_likelihood_instance=instance)

        search = al.m.MockSearch(name="test_search", samples=samples)

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        def modify_after_fit(
            paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
        ):
            pass

        analysis.modify_after_fit = modify_after_fit

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, ResultInterferometer)

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, interferometer_7, mask_2d_7x7
    ):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7,
            positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7
    ):
        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_via_instance_from(instance=instance)

        fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

        assert fit.log_likelihood == analysis_log_likelihood

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7
    ):
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(
            galaxies=af.Collection(lens=lens_galaxy),
            hyper_background_noise=hyper_background_noise,
        )

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_via_instance_from(instance=instance)

        fit = al.FitInterferometer(
            dataset=interferometer_7,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == analysis_log_likelihood

    def test__sets_up_hyper_galaxy_visibiltiies__froms(self, interferometer_7):

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): al.Array2D.ones(
                shape_native=(3, 3), pixel_scales=1.0
            ),
            ("galaxies", "source"): al.Array2D.full(
                fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        }

        hyper_galaxy_visibilities_path_dict = {
            ("galaxies", "lens"): al.Visibilities.full(fill_value=4.0, shape_slim=(7,)),
            ("galaxies", "source"): al.Visibilities.full(
                fill_value=5.0, shape_slim=(7,)
            ),
        }

        result = al.m.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=al.Visibilities.full(
                fill_value=6.0, shape_slim=(7,)
            ),
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7, hyper_dataset_result=result
        )

        analysis.set_hyper_dataset(result=result)

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "lens")].native
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].native
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.native == 3.0 * np.ones((3, 3))).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "lens")]
            == (4.0 + 4.0j) * np.ones((7,))
        ).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "source")]
            == (5.0 + 5.0j) * np.ones((7,))
        ).all()

        assert (analysis.hyper_model_visibilities == (6.0 + 6.0j) * np.ones((7,))).all()

    def test__stochastic_log_likelihoods_for_instance(self, interferometer_7):

        lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        source_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        source_hyper_image[4] = 10.0
        hyper_model_image = al.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): lens_hyper_image,
            ("galaxies", "source"): source_hyper_image,
        }

        result = al.m.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7,
            settings_lens=al.SettingsLens(stochastic_samples=2),
            hyper_dataset_result=result,
        )

        galaxies = af.ModelInstance()
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=5),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        log_evidences = analysis.stochastic_log_likelihoods_via_instance_from(
            instance=instance
        )

        assert len(log_evidences) == 2
        assert log_evidences[0] != log_evidences[1]

        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.DelaunayBrightnessImage(pixels=5),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        log_evidences = analysis.stochastic_log_likelihoods_via_instance_from(
            instance=instance
        )

        assert len(log_evidences) == 2
        assert log_evidences[0] != log_evidences[1]
