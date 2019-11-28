import autofit.optimize.non_linear.paths
import autolens as al
import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
from autolens.fit.fit import ImagingFit
from test_autolens.mock import mock_pipeline


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return al.Galaxy(
        redshift=1.0, light=al.lp.SphericalSersic(), mass=al.mp.SphericalIsothermal()
    )


@pytest.fixture(name="source_galaxy")
def make_source_galaxy():
    return al.Galaxy(redshift=2.0, light=al.lp.SphericalSersic())


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
def make_result(masked_imaging_7x7, instance):
    return al.PhaseImaging.Result(
        instance=instance,
        figure_of_merit=1.0,
        previous_model=af.ModelMapper(),
        gaussian_tuples=None,
        analysis=al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7, cosmology=cosmo.Planck15, image_path=""
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
        self.model = af.ModelMapper()
        self.mask = None
        self.positions = None


class MockAnalysis(object):
    pass


# noinspection PyAbstractClass
class MockOptimizer(af.NonLinearOptimizer):
    @af.convert_paths
    def __init__(self, paths):
        super().__init__(paths=paths)

    def fit(self, analysis, model):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.fit(None), None)


class MockPhase(object):
    def __init__(self):
        self.paths = autofit.optimize.non_linear.paths.Paths(
            phase_name="phase_name",
            phase_path="phase_path",
            phase_folders=("",),
            phase_tag="",
        )
        self.optimizer = MockOptimizer(paths=self.paths)
        self.model = af.ModelMapper()

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def run(self, *args, **kwargs):
        return MockResult()


class TestModelFixing(object):
    def test__defaults_both(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundBothPhase(MockPhase())

        instance = af.ModelInstance()
        instance.hyper_image_sky = al.hyper_data.HyperImageSky()
        instance.hyper_background_noise = al.hyper_data.HyperBackgroundNoise()

        mapper = phase.make_model(instance)

        assert isinstance(mapper.hyper_image_sky, af.PriorModel)
        assert isinstance(mapper.hyper_background_noise, af.PriorModel)

        assert mapper.hyper_image_sky.cls == al.hyper_data.HyperImageSky
        assert mapper.hyper_background_noise.cls == al.hyper_data.HyperBackgroundNoise

    def test__defaults_hyper_image_sky(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundSkyPhase(MockPhase())

        instance = af.ModelInstance()
        instance.hyper_image_sky = al.hyper_data.HyperImageSky()

        mapper = phase.make_model(instance)

        assert isinstance(mapper.hyper_image_sky, af.PriorModel)
        assert mapper.hyper_image_sky.cls == al.hyper_data.HyperImageSky

    def test__defaults_background_noise(self):
        # noinspection PyTypeChecker
        phase = al.InversionBackgroundNoisePhase(MockPhase())

        instance = af.ModelInstance()
        instance.hyper_background_noise = al.hyper_data.HyperBackgroundNoise()

        mapper = phase.make_model(instance)

        assert isinstance(mapper.hyper_background_noise, af.PriorModel)
        assert mapper.hyper_background_noise.cls == al.hyper_data.HyperBackgroundNoise

    def test__make_pixelization_model(self):
        instance = af.ModelInstance()
        mapper = af.ModelMapper()

        mapper.lens_galaxy = al.GalaxyModel(
            redshift=al.Redshift,
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
        )
        mapper.source_galaxy = al.GalaxyModel(
            redshift=al.Redshift, light=al.lp.EllipticalLightProfile
        )

        assert mapper.prior_count == 10

        instance.lens_galaxy = al.Galaxy(
            pixelization=al.pix.Rectangular(),
            regularization=al.reg.Constant(),
            redshift=1.0,
        )
        instance.source_galaxy = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalLightProfile()
        )

        # noinspection PyTypeChecker
        phase = al.ModelFixingHyperPhase(
            MockPhase(),
            "mock_phase",
            model_classes=(al.pix.Pixelization, al.reg.Regularization),
        )

        mapper = mapper.copy_with_fixed_priors(instance, phase.model_classes)

        assert mapper.prior_count == 3
        assert mapper.lens_galaxy.redshift == 1.0
        assert mapper.source_galaxy.light.axis_ratio == 1.0


class TestImagePassing(object):
    def test___image_dict(self, result):
        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "lens")].in_2d == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

    def test__results_are_passed_to_new_analysis__sets_up_hyper_images(
        self, mask_function_7x7, results_collection_7x7, imaging_7x7
    ):

        mask = mask_function_7x7(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=imaging_7x7.pixel_scales
        )

        results_collection_7x7[0].galaxy_images = [
            al.masked.array.full(fill_value=2.0, mask=mask),
            al.masked.array.full(fill_value=2.0, mask=mask),
        ]
        results_collection_7x7[0].galaxy_images[0][3] = -1.0
        results_collection_7x7[0].galaxy_images[1][5] = -1.0

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, hyper_galaxy=al.HyperGalaxy)
            ),
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, results=results_collection_7x7
        )

        assert (
            analysis.hyper_galaxy_image_path_dict[("g0",)].in_1d
            == np.array([2.0, 2.0, 2.0, 0.02, 2.0, 2.0, 2.0, 2.0, 2.0])
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("g1",)].in_1d
            == np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.02, 2.0, 2.0, 2.0])
        ).all()

        assert (
            analysis.hyper_model_image.in_1d
            == np.array([4.0, 4.0, 4.0, 2.02, 4.0, 2.02, 4.0, 4.0, 4.0])
        ).all()

    def test__associate_images_(self, instance, result, masked_imaging_7x7):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", result)
        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            cosmology=None,
            results=results_collection,
            image_path="",
        )

        instance = analysis.associate_images(instance=instance)

        lens_hyper_image = result.image_galaxy_dict[("galaxies", "lens")]
        source_hyper_image = result.image_galaxy_dict[("galaxies", "source")]

        hyper_model_image = lens_hyper_image + source_hyper_image

        assert instance.galaxies.lens.hyper_galaxy_image.in_2d == pytest.approx(
            lens_hyper_image.in_2d, 1.0e-4
        )
        assert instance.galaxies.source.hyper_galaxy_image.in_2d == pytest.approx(
            source_hyper_image.in_2d, 1.0e-4
        )

        assert instance.galaxies.lens.hyper_model_image.in_2d == pytest.approx(
            hyper_model_image.in_2d, 1.0e-4
        )
        assert instance.galaxies.source.hyper_model_image.in_2d == pytest.approx(
            hyper_model_image.in_2d, 1.0e-4
        )

    def test__fit_uses_hyper_fit_correctly(self, instance, result, masked_imaging_7x7):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", result)
        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            cosmology=cosmo.Planck15,
            results=results_collection,
            image_path="",
        )

        hyper_galaxy = al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        fit_figure_of_merit = analysis.fit(instance=instance)

        lens_hyper_image = result.image_galaxy_dict[("galaxies", "lens")]
        source_hyper_image = result.image_galaxy_dict[("galaxies", "source")]

        hyper_model_image = lens_hyper_image + source_hyper_image

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

        fit = ImagingFit(masked_imaging=masked_imaging_7x7, tracer=tracer)

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

    def test_combine_models(self, hyper_combined):
        result = MockResult()
        hyper_galaxy_result = MockResult()
        inversion_result = MockResult()

        hyper_galaxy_result.model = af.ModelMapper()
        inversion_result.model = af.ModelMapper()

        hyper_galaxy_result.model.hyper_galaxy = al.HyperGalaxy
        hyper_galaxy_result.model.pixelization = al.pix.Pixelization()
        inversion_result.model.pixelization = al.pix.Pixelization
        inversion_result.model.hyper_galaxy = al.HyperGalaxy()

        result.hyper_galaxy = hyper_galaxy_result
        result.inversion = inversion_result

        model = hyper_combined.combine_models(result)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert isinstance(model.pixelization, af.PriorModel)

        assert model.hyper_galaxy.cls == al.HyperGalaxy
        assert model.pixelization.cls == al.pix.Pixelization

    def test_instantiation(self, hyper_combined):
        assert len(hyper_combined.hyper_phases) == 2

        galaxy_phase = hyper_combined.hyper_phases[0]
        pixelization_phase = hyper_combined.hyper_phases[1]

        assert galaxy_phase.hyper_name == "hyper_galaxy"
        assert isinstance(galaxy_phase, al.HyperGalaxyPhase)

        assert pixelization_phase.hyper_name == "inversion"
        assert isinstance(pixelization_phase, al.InversionPhase)

    def test_hyper_result(self, imaging_7x7):
        normal_phase = MockPhase()

        # noinspection PyTypeChecker
        phase = al.HyperGalaxyPhase(normal_phase)

        # noinspection PyUnusedLocal
        def run_hyper(*args, **kwargs):
            return MockResult()

        phase.run_hyper = run_hyper

        result = phase.run(imaging_7x7)

        assert hasattr(result, "hyper_galaxy")
        assert isinstance(result.hyper_galaxy, MockResult)


class TestHyperGalaxyPhase(object):
    def test__likelihood_function_is_same_as_normal_phase_likelihood_function(
        self, imaging_7x7, mask_function_7x7
    ):

        hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
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

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7)
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])

        mask = phase_imaging_7x7.meta_imaging_fit.setup_phase_mask(
            shape_2d=imaging_7x7.shape_2d,
            pixel_scales=imaging_7x7.pixel_scales,
            mask=None,
        )
        assert mask.sub_size == 2

        masked_imaging = al.masked.imaging(imaging=imaging_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = ImagingFit(
            masked_imaging=masked_imaging,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        phase_imaging_7x7_hyper = phase_imaging_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True
        )

        instance = phase_imaging_7x7_hyper.model.instance_from_unit_vector([])

        instance.hyper_galaxy = al.HyperGalaxy(noise_factor=0.0)

        analysis = phase_imaging_7x7_hyper.hyper_phases[0].Analysis(
            masked_imaging=masked_imaging,
            hyper_model_image=fit.model_image,
            hyper_galaxy_image=fit.model_image,
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
