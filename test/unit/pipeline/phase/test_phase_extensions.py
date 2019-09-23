import autolens as al
import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
from test.unit.mock.pipeline import mock_pipeline


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(),
        mass=al.mass_profiles.SphericalIsothermal(),
    )


@pytest.fixture(name="source_galaxy")
def make_source_galaxy():
    return al.Galaxy(redshift=2.0, light=al.light_profiles.SphericalSersic())


@pytest.fixture(name="all_galaxies")
def make_all_galaxies(lens_galaxy, source_galaxy):
    galaxies = af.ModelInstance()
    galaxies.lens = lens_galaxy
    galaxies.source = source_galaxy
    return galaxies


@pytest.fixture(name="instance")
def make_instance(all_galaxies):
    instance = af.ModelInstance()
    instance.galaxies = all_galaxies
    return instance


@pytest.fixture(name="result")
def make_result(lens_imaging_data_7x7, instance):
    return al.PhaseImaging.Result(
        constant=instance,
        figure_of_merit=1.0,
        previous_variable=af.ModelMapper(),
        gaussian_tuples=None,
        analysis=al.PhaseImaging.Analysis(
            lens_imaging_data=lens_imaging_data_7x7,
            cosmology=cosmo.Planck15,
            image_path="",
        ),
        optimizer=None,
    )


class MostLikelyFit(object):
    def __init__(self, model_image_2d):
        self.model_image_2d = model_image_2d


class MockResult(object):
    def __init__(self, most_likely_fit=None):
        self.most_likely_fit = most_likely_fit
        self.analysis = MockAnalysis()
        self.path_galaxy_tuples = []
        self.variable = af.ModelMapper()
        self.mask = None
        self.positions = None


class MockAnalysis(object):
    pass


# noinspection PyAbstractClass
class MockOptimizer(af.NonLinearOptimizer):
    def __init__(
        self,
        phase_name="mock_optimizer",
        phase_tag="tag",
        phase_folders=tuple(),
        model_mapper=None,
    ):
        super().__init__(
            phase_folders=phase_folders,
            phase_tag=phase_tag,
            phase_name=phase_name,
            model_mapper=model_mapper,
        )

    def fit(self, analysis):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.fit(None), None)


class MockPhase(object):
    def __init__(self):
        self.phase_name = "phase name"
        self.phase_path = "phase_path"
        self.optimizer = MockOptimizer()
        self.phase_folders = [""]
        self.phase_tag = ""

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def run(self, *args, **kwargs):
        return MockResult()


class TestVariableFixing(object):
    def test_defaults_both(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundBothPhase(MockPhase())

        constant = af.ModelInstance()
        constant.hyper_image_sky = al.HyperImageSky()
        constant.hyper_background_noise = al.HyperBackgroundNoise()

        mapper = phase.make_variable(constant)

        assert isinstance(mapper.hyper_image_sky, af.PriorModel)
        assert isinstance(mapper.hyper_background_noise, af.PriorModel)

        assert mapper.hyper_image_sky.cls == al.HyperImageSky
        assert mapper.hyper_background_noise.cls == al.HyperBackgroundNoise

    def test_defaults_hyper_image_sky(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundSkyPhase(MockPhase())

        constant = af.ModelInstance()
        constant.hyper_image_sky = al.HyperImageSky()

        mapper = phase.make_variable(constant)

        assert isinstance(mapper.hyper_image_sky, af.PriorModel)
        assert mapper.hyper_image_sky.cls == al.HyperImageSky

    def test_defaults_background_noise(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundNoisePhase(MockPhase())

        constant = af.ModelInstance()
        constant.hyper_background_noise = al.HyperBackgroundNoise()

        mapper = phase.make_variable(constant)

        assert isinstance(mapper.hyper_background_noise, af.PriorModel)
        assert mapper.hyper_background_noise.cls == al.HyperBackgroundNoise

    def test_make_pixelization_variable(self):
        instance = af.ModelInstance()
        mapper = af.ModelMapper()

        mapper.lens_galaxy = al.GalaxyModel(
            redshift=al.Redshift,
            pixelization=al.pixelizations.Rectangular,
            regularization=al.regularization.Constant,
        )
        mapper.source_galaxy = al.GalaxyModel(
            redshift=al.Redshift, light=al.light_profiles.EllipticalLightProfile
        )

        assert mapper.prior_count == 9

        instance.lens_galaxy = al.Galaxy(
            pixelization=al.pixelizations.Rectangular(),
            regularization=al.regularization.Constant(),
            redshift=1.0,
        )
        instance.source_galaxy = al.Galaxy(
            redshift=1.0, light=al.light_profiles.EllipticalLightProfile()
        )

        # noinspection PyTypeChecker
        phase = al.VariableFixingHyperPhase(
            MockPhase(),
            "mock_phase",
            variable_classes=(
                al.pixelizations.Pixelization,
                al.regularization.Regularization,
            ),
        )

        mapper = mapper.copy_with_fixed_priors(instance, phase.variable_classes)

        assert mapper.prior_count == 3
        assert mapper.lens_galaxy.redshift == 1.0
        assert mapper.source_galaxy.light.axis_ratio == 1.0


class TestImagePassing(object):
    def test__image_dict(self, result):
        image_dict = result.image_galaxy_2d_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.constant.galaxies.lens = al.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_2d_dict
        assert (image_dict[("galaxies", "lens")] == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

    def test_galaxy_image_dict(
        self, lens_galaxy, source_galaxy, sub_grid_7x7, convolver_7x7, blurring_grid_7x7
    ):
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        assert (
            len(
                tracer.galaxy_blurred_profile_image_dict_from_grid_and_convolver(
                    grid=sub_grid_7x7,
                    convolver=convolver_7x7,
                    blurring_grid=blurring_grid_7x7,
                )
            )
            == 2
        )
        assert (
            lens_galaxy
            in tracer.galaxy_blurred_profile_image_dict_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )
        )
        assert (
            source_galaxy
            in tracer.galaxy_blurred_profile_image_dict_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )
        )

    def test__results_are_passed_to_new_analysis__sets_up_hyper_images(
        self, mask_function_7x7, results_collection_7x7, imaging_data_7x7
    ):
        results_collection_7x7[0].galaxy_images = [
            2.0 * np.ones((7, 7)),
            2.0 * np.ones((7, 7)),
        ]
        results_collection_7x7[0].galaxy_images[0][3, 2] = -1.0
        results_collection_7x7[0].galaxy_images[1][3, 4] = -1.0

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, hyper_galaxy=al.HyperGalaxy)
            ),
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (
            analysis.hyper_galaxy_image_1d_path_dict[("g0",)]
            == np.array([2.0, 2.0, 2.0, 0.02, 2.0, 2.0, 2.0, 2.0, 2.0])
        ).all()

        assert (
            analysis.hyper_galaxy_image_1d_path_dict[("g1",)]
            == np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.02, 2.0, 2.0, 2.0])
        ).all()

        assert (
            analysis.hyper_model_image_1d
            == np.array([4.0, 4.0, 4.0, 2.02, 4.0, 2.02, 4.0, 4.0, 4.0])
        ).all()

    def test__results_are_passed_to_new_analysis__hyper_images_values_below_minimum_are_scaled_up_using_config(
        self, mask_function_7x7, results_collection_7x7, imaging_data_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, hyper_galaxy=al.HyperGalaxy)
            ),
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (analysis.hyper_model_image_1d == 5.0 * np.ones(9)).all()

        assert (
            analysis.hyper_galaxy_image_1d_path_dict[("g0",)] == 2.0 * np.ones(9)
        ).all()
        assert (
            analysis.hyper_galaxy_image_1d_path_dict[("g1",)] == 3.0 * np.ones(9)
        ).all()

    def test__results_are_passed_to_new_analysis__sets_up_hyper_cluster_images__includes_hyper_minimum(
        self, mask_function_7x7, results_collection_7x7, imaging_data_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    hyper_galaxy=al.HyperGalaxy,
                    pixelization=al.pixelizations.VoronoiBrightnessImage,
                    regularization=al.regularization.Constant,
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=5,
            pixel_scale_binned_cluster_grid=None,
            optimizer_class=mock_pipeline.MockNLO,
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)] == 2.0 * np.ones(9)
        ).all()
        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)] == 3.0 * np.ones(9)
        ).all()

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    hyper_galaxy=al.HyperGalaxy,
                    pixelization=al.pixelizations.VoronoiBrightnessImage,
                    regularization=al.regularization.Constant,
                )
            ),
            inversion_pixel_limit=1,
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            pixel_scale_binned_cluster_grid=imaging_data_7x7.pixel_scale,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)] == 2.0 * np.ones(9)
        ).all()
        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)] == 3.0 * np.ones(9)
        ).all()
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)])
            == analysis.lens_data.grid.binned.shape[0]
        )
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)])
            == analysis.lens_data.grid.binned.shape[0]
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    hyper_galaxy=al.HyperGalaxy,
                    pixelization=al.pixelizations.VoronoiBrightnessImage,
                    regularization=al.regularization.Constant,
                )
            ),
            inversion_pixel_limit=1,
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            pixel_scale_binned_cluster_grid=imaging_data_7x7.pixel_scale * 2.0,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)]
            == np.array([0.5, 1.0, 1.0, 2.0])
        ).all()
        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)]
            == np.array([0.75, 1.5, 1.5, 3.0])
        ).all()
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)])
            == analysis.lens_data.grid.binned.shape[0]
        )
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)])
            == analysis.lens_data.grid.binned.shape[0]
        )

        results_collection_7x7[0].galaxy_images = [
            2.0 * np.ones((7, 7)),
            2.0 * np.ones((7, 7)),
        ]
        results_collection_7x7[0].galaxy_images[0][3, 2] = -1.0
        results_collection_7x7[0].galaxy_images[1][3, 4] = -1.0

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    hyper_galaxy=al.HyperGalaxy,
                    pixelization=al.pixelizations.VoronoiBrightnessImage,
                    regularization=al.regularization.Constant,
                )
            ),
            inversion_pixel_limit=1,
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            pixel_scale_binned_cluster_grid=imaging_data_7x7.pixel_scale * 2.0,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)]
            == np.array([2.0, 2.0, 1.25, 2.0])
        ).all()
        assert (
            analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)]
            == np.array([2.0, 2.0, 2.0, 1.25])
        ).all()
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g0",)])
            == analysis.lens_data.grid.binned.shape[0]
        )
        assert (
            len(analysis.binned_hyper_galaxy_image_1d_path_dict[("g1",)])
            == analysis.lens_data.grid.binned.shape[0]
        )

    def test__associate_images_(self, instance, result, lens_imaging_data_7x7):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", result)
        analysis = al.PhaseImaging.Analysis(
            lens_imaging_data=lens_imaging_data_7x7,
            cosmology=None,
            results=results_collection,
            image_path="",
        )

        instance = analysis.associate_images(instance=instance)

        hyper_lens_image_1d = lens_imaging_data_7x7.mapping.array_1d_from_array_2d(
            array_2d=result.image_galaxy_2d_dict[("galaxies", "lens")]
        )
        hyper_source_image_1d = lens_imaging_data_7x7.mapping.array_1d_from_array_2d(
            array_2d=result.image_galaxy_2d_dict[("galaxies", "source")]
        )

        hyper_model_image_1d = hyper_lens_image_1d + hyper_source_image_1d

        assert instance.galaxies.lens.hyper_galaxy_image_1d == pytest.approx(
            hyper_lens_image_1d, 1.0e-4
        )
        assert instance.galaxies.source.hyper_galaxy_image_1d == pytest.approx(
            hyper_source_image_1d, 1.0e-4
        )

        assert instance.galaxies.lens.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4
        )
        assert instance.galaxies.source.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4
        )

    def test__fit_uses_hyper_fit_correctly_(
        self, instance, result, lens_imaging_data_7x7
    ):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", result)
        analysis = al.PhaseImaging.Analysis(
            lens_imaging_data=lens_imaging_data_7x7,
            cosmology=cosmo.Planck15,
            results=results_collection,
            image_path="",
        )

        hyper_galaxy = al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        fit_figure_of_merit = analysis.fit(instance=instance)

        hyper_lens_image_1d = lens_imaging_data_7x7.mapping.array_1d_from_array_2d(
            array_2d=result.image_galaxy_2d_dict[("galaxies", "lens")]
        )
        hyper_source_image_1d = lens_imaging_data_7x7.mapping.array_1d_from_array_2d(
            array_2d=result.image_galaxy_2d_dict[("galaxies", "source")]
        )

        hyper_model_image_1d = hyper_lens_image_1d + hyper_source_image_1d

        g0 = al.Galaxy(
            redshift=0.5,
            light_profile=instance.galaxies.lens.light,
            mass_profile=instance.galaxies.lens.mass,
            hyper_galaxy=hyper_galaxy,
            hyper_model_image_1d=hyper_model_image_1d,
            hyper_galaxy_image_1d=hyper_lens_image_1d,
            hyper_minimum_value=0.0,
        )
        g1 = al.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        fit = al.LensImagingFit.from_lens_data_and_tracer(
            lens_data=lens_imaging_data_7x7, tracer=tracer
        )

        assert (fit_figure_of_merit == fit.figure_of_merit).all()


@pytest.fixture(name="hyper_combined")
def make_combined():
    normal_phase = MockPhase()

    # noinspection PyUnusedLocal
    def run_hyper(*args, **kwargs):
        return MockResult()

    # noinspection PyTypeChecker
    hyper_combined = al.CombinedHyperPhase(
        normal_phase, hyper_phase_classes=(al.HyperGalaxyPhase, al.InversionPhase)
    )

    for phase in hyper_combined.hyper_phases:
        phase.run_hyper = run_hyper

    return hyper_combined


class TestHyperAPI(object):
    def test_combined_result(self, hyper_combined):
        result = hyper_combined.run(None)

        assert hasattr(result, "hyper_galaxy")
        assert isinstance(result.hyper_galaxy, MockResult)

        assert hasattr(result, "inversion")
        assert isinstance(result.inversion, MockResult)

        assert hasattr(result, "hyper_combined")
        assert isinstance(result.hyper_combined, MockResult)

    def test_combine_variables(self, hyper_combined):
        result = MockResult()
        hyper_galaxy_result = MockResult()
        inversion_result = MockResult()

        hyper_galaxy_result.variable = af.ModelMapper()
        inversion_result.variable = af.ModelMapper()

        hyper_galaxy_result.variable.hyper_galaxy = al.HyperGalaxy
        hyper_galaxy_result.variable.pixelization = al.pixelizations.Pixelization()
        inversion_result.variable.pixelization = al.pixelizations.Pixelization
        inversion_result.variable.hyper_galaxy = al.HyperGalaxy()

        result.hyper_galaxy = hyper_galaxy_result
        result.inversion = inversion_result

        variable = hyper_combined.combine_variables(result)

        assert isinstance(variable.hyper_galaxy, af.PriorModel)
        assert isinstance(variable.pixelization, af.PriorModel)

        assert variable.hyper_galaxy.cls == al.HyperGalaxy
        assert variable.pixelization.cls == al.pixelizations.Pixelization

    def test_instantiation(self, hyper_combined):
        assert len(hyper_combined.hyper_phases) == 2

        galaxy_phase = hyper_combined.hyper_phases[0]
        pixelization_phase = hyper_combined.hyper_phases[1]

        assert galaxy_phase.hyper_name == "hyper_galaxy"
        assert isinstance(galaxy_phase, al.HyperGalaxyPhase)

        assert pixelization_phase.hyper_name == "inversion"
        assert isinstance(pixelization_phase, al.InversionPhase)

    def test_hyper_result(self, imaging_data_7x7):
        normal_phase = MockPhase()

        # noinspection PyTypeChecker
        phase = al.HyperGalaxyPhase(normal_phase)

        # noinspection PyUnusedLocal
        def run_hyper(*args, **kwargs):
            return MockResult()

        phase.run_hyper = run_hyper

        result = phase.run(imaging_data_7x7)

        assert hasattr(result, "hyper_galaxy")
        assert isinstance(result.hyper_galaxy, MockResult)


class TestHyperGalaxyPhase(object):
    def test__likelihood_function_is_same_as_normal_phase_likelihood_function(
        self, imaging_data_7x7, mask_function_7x7
    ):

        hyper_image_sky = al.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=dict(lens=lens_galaxy),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            sub_size=2,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])

        mask = phase_imaging_7x7.mask_function(image=imaging_data_7x7.image, sub_size=2)
        lens_data = al.LensImagingData(imaging_data=imaging_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensImagingFit.from_lens_data_and_tracer(
            lens_data=lens_data,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        phase_imaging_7x7_hyper = phase_imaging_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True
        )

        instance = phase_imaging_7x7_hyper.variable.instance_from_unit_vector([])

        instance.hyper_galaxy = al.HyperGalaxy(noise_factor=0.0)

        analysis = phase_imaging_7x7_hyper.hyper_phases[0].Analysis(
            lens_data=lens_data,
            hyper_model_image_1d=fit.model_image(return_in_2d=False),
            hyper_galaxy_image_1d=fit.model_image(return_in_2d=False),
            image_path=None,
        )

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=al.HyperGalaxy(noise_factor=0.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.figure_of_merit == fit.figure_of_merit

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=al.HyperGalaxy(noise_factor=1.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.figure_of_merit != fit.figure_of_merit

        instance.hyper_galaxy = al.HyperGalaxy(noise_factor=0.0)

        figure_of_merit = analysis.fit(instance=instance)

        assert figure_of_merit == fit.figure_of_merit
