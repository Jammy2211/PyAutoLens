from autolens.pipeline import pipeline as pl
from autolens.pipeline import profile_pipeline
from autolens.pipeline import phase as ph
from autolens.autopipe import model_mapper
from autolens.autopipe import non_linear
from autolens.imaging import image as im
from autolens.profiles import light_profiles
from autolens.profiles import mass_profiles
from autolens.analysis import galaxy_prior as gp
from autolens.analysis import galaxy as g
import numpy as np
import pytest


class DummyPhase(object):
    def __init__(self):
        self.masked_image = None
        self.previous_results = None
        self.name = "dummy_phase"

    def run(self, masked_image, previous_results):
        self.masked_image = masked_image
        self.previous_results = previous_results
        return non_linear.Result(model_mapper.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        pipeline = pl.Pipeline("", phase_1, phase_2)

        pipeline.run(None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        phase_3 = DummyPhase()

        pipeline1 = pl.Pipeline("", phase_1, phase_2)
        pipeline2 = pl.Pipeline("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases


shape = (50, 50)


@pytest.fixture(name="profile_only_pipeline")
def make_profile_only_pipeline():
    return profile_pipeline.make()


@pytest.fixture(name="image")
def make_image():
    return im.Image(np.ones(shape), pixel_scale=0.2, noise=np.ones(shape), psf=im.PSF(np.ones((3, 3))))


@pytest.fixture(name="results_1")
def make_results_1():
    const = model_mapper.ModelInstance()
    var = model_mapper.ModelMapper()
    const.lens_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersic())
    var.lens_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)
    return ph.SourceLensPhase.Result(const, 1, var, [np.full(shape, 0.5), None])


@pytest.fixture(name="results_2")
def make_results_2():
    const = model_mapper.ModelInstance()
    var = model_mapper.ModelMapper()
    var.lens_galaxy = gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermal)
    var.source_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)
    const.lens_galaxy = g.Galaxy(sie=mass_profiles.SphericalIsothermal())
    const.source_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersic())
    return ph.SourceLensPhase.Result(const, 1, var, [np.full(shape, 0.5), np.full(shape, 0.5)])


@pytest.fixture(name="results_3")
def make_results_3():
    const = model_mapper.ModelInstance()
    var = model_mapper.ModelMapper()
    var.lens_galaxy = gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermal,
                                     elliptical_sersic=light_profiles.EllipticalSersic)
    var.source_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)
    const.lens_galaxy = g.Galaxy(sie=mass_profiles.SphericalIsothermal(),
                                 elliptical_sersic=light_profiles.EllipticalSersic())
    const.source_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersic())
    return ph.SourceLensPhase.Result(const, 1, var, [np.full(shape, 0.5), np.full(shape, 0.5)])


@pytest.fixture(name="results_3h")
def make_results_3h():
    const = model_mapper.ModelInstance()
    var = model_mapper.ModelMapper()
    const.lens_galaxy = g.Galaxy(hyper_galaxy=g.HyperGalaxy())
    const.source_galaxy = g.Galaxy(hyper_galaxy=g.HyperGalaxy())
    return ph.SourceLensPhase.Result(const, 1, var, [np.full(shape, 0.5), np.full(shape, 0.5)])


class TestProfileOnlyPipeline(object):
    def test_phase1(self, profile_only_pipeline, image):
        phase1 = profile_only_pipeline.phases[0]
        analysis = phase1.make_analysis(image)

        assert isinstance(phase1.lens_galaxy, gp.GalaxyPrior)
        assert phase1.source_galaxy is None

        assert analysis.masked_image == np.ones((716,))
        assert analysis.masked_image.sub_grid_size == 1
        assert analysis.previous_results is None

    def test_phase2(self, profile_only_pipeline, image, results_1):
        phase2 = profile_only_pipeline.phases[1]
        previous_results = ph.ResultsCollection([results_1])
        analysis = phase2.make_analysis(image, previous_results)

        assert analysis.masked_image == np.full((704,), 0.5)

        assert isinstance(phase2.lens_galaxy, gp.GalaxyPrior)
        assert isinstance(phase2.source_galaxy, gp.GalaxyPrior)
        assert phase2.lens_galaxy.sie.centre == previous_results.first.variable.lens_galaxy.elliptical_sersic.centre

    def test_phase3(self, profile_only_pipeline, image, results_1, results_2):
        phase3 = profile_only_pipeline.phases[2]
        previous_results = ph.ResultsCollection([results_1, results_2])

        analysis = phase3.make_analysis(image, previous_results)

        assert isinstance(phase3.lens_galaxy, gp.GalaxyPrior)
        assert isinstance(phase3.source_galaxy, gp.GalaxyPrior)

        assert analysis.masked_image == np.ones((716,))

        assert phase3.lens_galaxy.elliptical_sersic == results_1.variable.lens_galaxy.elliptical_sersic
        assert phase3.lens_galaxy.sie == results_2.variable.lens_galaxy.sie
        assert phase3.source_galaxy == results_2.variable.source_galaxy

    def test_phase3h(self, profile_only_pipeline, image, results_1, results_2, results_3):
        phase3h = profile_only_pipeline.phases[3]
        previous_results = ph.ResultsCollection([results_1, results_2, results_3])

        analysis = phase3h.make_analysis(image, previous_results)

        assert isinstance(phase3h.lens_galaxy, gp.GalaxyPrior)
        assert isinstance(phase3h.source_galaxy, gp.GalaxyPrior)

        assert analysis.masked_image == np.ones((716,))

        assert phase3h.lens_galaxy.elliptical_sersic == results_3.constant.lens_galaxy.elliptical_sersic
        assert phase3h.lens_galaxy.sie == results_3.constant.lens_galaxy.sie
        assert phase3h.source_galaxy.elliptical_sersic == results_3.constant.source_galaxy.elliptical_sersic

        assert isinstance(phase3h.lens_galaxy.hyper_galaxy, model_mapper.PriorModel)
        assert isinstance(phase3h.source_galaxy.hyper_galaxy, model_mapper.PriorModel)

    def test_phase4(self, profile_only_pipeline, image, results_1, results_2, results_3, results_3h):
        phase4 = profile_only_pipeline.phases[4]
        previous_results = ph.ResultsCollection([results_1, results_2, results_3, results_3h])

        analysis = phase4.make_analysis(image, previous_results)

        assert isinstance(phase4.lens_galaxy, gp.GalaxyPrior)
        assert isinstance(phase4.source_galaxy, gp.GalaxyPrior)

        assert analysis.masked_image == np.ones((716,))

        assert isinstance(phase4.lens_galaxy.hyper_galaxy, g.HyperGalaxy)

        assert phase4.lens_galaxy.hyper_galaxy == results_3h.constant.lens_galaxy.hyper_galaxy
        assert phase4.source_galaxy.hyper_galaxy == results_3h.constant.source_galaxy.hyper_galaxy

        assert phase4.lens_galaxy.elliptical_sersic == results_3.variable.lens_galaxy.elliptical_sersic
        assert phase4.lens_galaxy.sie == results_3.variable.lens_galaxy.sie
        assert phase4.source_galaxy.elliptical_sersic == results_3.variable.source_galaxy.elliptical_sersic
