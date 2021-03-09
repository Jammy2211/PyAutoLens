from os import path

import autofit as af
import autolens as al
from autolens import exc
import pytest
from autolens.analysis import result as res
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestAnalysisAbstract:
    def test__auto_einstein_radius_is_used__einstein_radius_used_in_analysis(
        self, masked_imaging_7x7
    ):

        settings_lens = al.SettingsLens(auto_einstein_radius_factor=None)

        tracer = mock.MockTracer(einstein_radius=2.0)

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            results=mock.MockResults(max_log_likelihood_tracer=tracer),
            settings_lens=settings_lens,
        )

        assert analysis.settings_lens.einstein_radius_estimate == None

        settings_lens = al.SettingsLens(auto_einstein_radius_factor=1.0)

        tracer = mock.MockTracer(einstein_radius=2.0)

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            results=mock.MockResults(max_log_likelihood_tracer=tracer),
            settings_lens=settings_lens,
        )

        assert analysis.settings_lens.einstein_radius_estimate == 2.0


class TestAnalysisDataset:
    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, masked_imaging_7x7
    ):

        masked_imaging_7x7.imaging.positions = al.Grid2DIrregular(
            [(1.0, 100.0), (200.0, 2.0)]
        )

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        settings_lens = al.SettingsLens(positions_threshold=0.01)

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7, settings_lens=settings_lens
        )
        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)


class TestAnalysisImaging:
    def test__make_result__result_imaging_is_returned(self, masked_imaging_7x7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=al.Galaxy(redshift=0.5))
        )

        search = mock.MockSearch(name="test_phase")

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImaging)

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, masked_imaging_7x7
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        masked_imaging_7x7.imaging.positions = al.Grid2DIrregular(
            [(1.0, 100.0), (200.0, 2.0)]
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, masked_imaging_7x7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(lens=lens_galaxy)
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitImaging(masked_imaging=masked_imaging_7x7, tracer=tracer)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, masked_imaging_7x7
    ):

        hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        model = af.CollectionPriorModel(
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            galaxies=af.CollectionPriorModel(lens=lens_galaxy),
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.FitImaging(
            masked_imaging=masked_imaging_7x7,
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

        lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        hyper_model_image = al.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "lens"): lens_hyper_image}

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, results=results)

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

        fit = al.FitImaging(masked_imaging=masked_imaging_7x7, tracer=tracer)

        assert (fit.tracer.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
        assert fit_likelihood == fit.log_likelihood

    def test__stochastic_log_evidences_for_instance(self, masked_imaging_7x7):

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

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiMagnification(shape=(3, 3)),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, results=results)

        stochastic_log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert stochastic_log_evidences == None

        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=9),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, results=results)

        stochastic_log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert stochastic_log_evidences[0] != stochastic_log_evidences[1]


class TestAnalysisInterferometer:
    def test__make_result__result_interferometer_is_returned(
        self, masked_interferometer_7
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=al.Galaxy(redshift=0.5))
        )

        search = mock.MockSearch(name="test_phase")

        analysis = al.AnalysisInterferometer(dataset=masked_interferometer_7)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultInterferometer)

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, interferometer_7, mask_7x7, visibilities_mask_7
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        interferometer_7.positions = al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)])

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7,
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, masked_interferometer_7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(lens=lens_galaxy)
        )

        analysis = al.AnalysisInterferometer(dataset=masked_interferometer_7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitInterferometer(
            masked_interferometer=masked_interferometer_7, tracer=tracer
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, masked_interferometer_7
    ):
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(lens=lens_galaxy),
            hyper_background_noise=hyper_background_noise,
        )

        analysis = al.AnalysisInterferometer(dataset=masked_interferometer_7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitInterferometer(
            masked_interferometer=masked_interferometer_7,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__stochastic_log_evidences_for_instance(self, masked_interferometer_7):

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

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.AnalysisInterferometer(
            dataset=masked_interferometer_7,
            settings_lens=al.SettingsLens(stochastic_samples=2),
            results=results,
        )

        log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert len(log_evidences) == 2
        assert log_evidences[0] != log_evidences[1]
