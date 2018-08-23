from autolens.pipeline import phase as ph
import pytest
from autolens.analysis import galaxy as g
from autolens.analysis import galaxy_prior as gp
from autolens.autopipe import non_linear
import numpy as np
from autolens.imaging import mask as msk
from autolens.imaging import image as img
from autolens.imaging import masked_image as mi
from autolens.autopipe import model_mapper as mm
from autolens.profiles import light_profiles, mass_profiles
from autolens import conf
from os import path
import os

directory = path.dirname(path.realpath(__file__))

shape = (10, 10)


class MockAnalysis(object):

    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # def tracer_for_instance(self, instance):
    #     from autolens.analysis import ray_tracing
    #     return ray_tracing.Tracer(lens_galaxies=[], source_galaxies=[], image_plane_grids=)

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]


class MockResults(object):
    def __init__(self, model_image, galaxy_images=()):
        self.model_image = model_image
        self.galaxy_images = galaxy_images
        self.constant = mm.ModelInstance()
        self.variable = mm.ModelMapper()


class NLO(non_linear.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector, constant):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = constant

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)
                for key, value in self.constant.__dict__.items():
                    setattr(instance, key, value)

                likelihood = analysis.fit(instance)
                self.result = non_linear.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.total_parameters * [0.5])

        return fitness_function.result


@pytest.fixture(name="grids")
def make_grids(masked_image):
    return msk.GridCollection.from_mask_sub_grid_size_and_blurring_shape(
        masked_image.mask, 1, masked_image.psf.shape)


@pytest.fixture(name="phase")
def make_phase():
    return ph.LensMassAndSourceProfilePhase(optimizer_class=NLO)


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_prior")
def make_galaxy_prior():
    return gp.GalaxyPrior()


@pytest.fixture(name="image")
def make_image():
    image = img.Image(np.array(np.zeros(shape)), pixel_scale=1.0, psf=img.PSF(np.ones((3, 3))), noise=np.ones(shape))
    return image


@pytest.fixture(name="masked_image")
def make_masked_image():
    image = img.Image(np.array(np.zeros(shape)), pixel_scale=1.0, psf=img.PSF(np.ones((3, 3))), noise=np.ones(shape))
    mask = msk.Mask.circular(shape, 1, 3)
    return mi.MaskedImage(image, mask)


@pytest.fixture(name="results")
def make_results():
    return MockResults(np.ones((10, 10)),
                       galaxy_images=[np.ones((10, 10)), np.ones((10, 10))])


@pytest.fixture(name="results_collection")
def make_results_collection(results):
    return ph.ResultsCollection([results])


def clean_images():
    try:
        os.remove('{}/source_lens_phase/source_image_0.fits'.format(directory))
        os.remove('{}/source_lens_phase/lens_image_0.fits'.format(directory))
        os.remove('{}/source_lens_phase/model_image_0.fits'.format(directory))
    except FileNotFoundError:
        pass
    conf.instance.data_path = directory


class TestPhase(object):

    def test_set_constants(self, phase, galaxy):
        phase.lens_galaxies = [galaxy]
        assert phase.optimizer.constant.lens_galaxies == [galaxy]
        assert phase.optimizer.variable.lens_galaxies == []

    def test_set_variables(self, phase, galaxy_prior):
        phase.lens_galaxies = [galaxy_prior]
        assert phase.optimizer.variable.lens_galaxies == [galaxy_prior]
        assert phase.optimizer.constant.lens_galaxies == []

    def test_mask_analysis(self, phase, image, masked_image):
        analysis = phase.make_analysis(image=image)
        assert analysis.last_results is None
        assert analysis.masked_image == masked_image

    def test_fit(self, image):
        clean_images()
        phase = ph.LensMassAndSourceProfilePhase(optimizer_class=NLO,
                                                 lens_galaxies=[g.Galaxy()], source_galaxies=[g.Galaxy()])
        result = phase.run(image=image)
        assert isinstance(result.constant.lens_galaxies[0], g.Galaxy)
        assert isinstance(result.constant.source_galaxies[0], g.Galaxy)

    def test_customize(self, results, image):
        class MyPlanePhaseAnd(ph.LensMassAndSourceProfilePhase):
            def pass_priors(self, previous_results):
                self.lens_galaxies = previous_results.last.constant.lens_galaxies
                self.source_galaxies = previous_results.last.variable.source_galaxies

        galaxy = g.Galaxy()
        galaxy_prior = gp.GalaxyPrior()

        setattr(results.constant, "lens_galaxies", [galaxy])
        setattr(results.variable, "source_galaxies", [galaxy_prior])

        phase = MyPlanePhaseAnd(optimizer_class=NLO)
        phase.make_analysis(image=image, previous_results=ph.ResultsCollection([results]))

        assert phase.lens_galaxies == [galaxy]
        assert phase.source_galaxies == [galaxy_prior]

    def test_default_mask_function(self, phase, image):
        assert len(mi.MaskedImage(image, phase.mask_function(image))) == 32

    # TODO: removed because galaxy_images seems to have been removed?
    # def test_galaxy_images(self, image, phase):
    #     clean_images()
    #     phase.lens_galaxies = [g.Galaxy()]
    #     phase.source_galaxies = [g.Galaxy()]
    #     result = phase.run(image)
    #     assert len(result.galaxy_images) == 2

    def test_duplication(self):
        phase = ph.LensMassAndSourceProfilePhase(lens_galaxies=[gp.GalaxyPrior()], source_galaxies=[gp.GalaxyPrior()])

        ph.LensMassAndSourceProfilePhase()

        assert phase.lens_galaxies is not None
        assert phase.source_galaxies is not None

    def test_modify_image(self, image):
        class MyPhase(ph.Phase):
            def modify_image(self, im, previous_results):
                assert image.shape == im.shape
                return im

        phase = MyPhase(phase_name='phase')
        analysis = phase.make_analysis(image)
        assert analysis.masked_image != image

    def test_tracer_for_instance(self, image):
        lens_galaxy = g.Galaxy()
        source_galaxy = g.Galaxy()
        phase = ph.LensMassAndSourceProfilePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])
        analysis = phase.make_analysis(image)
        instance = phase.constant
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.source_plane.galaxies[0] == source_galaxy

    def test__phase_can_receive_list_of_galaxy_priors(self):
        phase = ph.LensProfilePhase(lens_galaxies=[gp.GalaxyPrior(sersic=light_profiles.EllipticalSersicLP,
                                                                  sis=mass_profiles.SphericalIsothermalMP,
                                                                  variable_redshift=True),
                                                   gp.GalaxyPrior(sis=mass_profiles.SphericalIsothermalMP,
                                                                  variable_redshift=True)],
                                    optimizer_class=non_linear.MultiNest)

        instance = phase.optimizer.variable.instance_from_physical_vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8])

        assert instance.lens_galaxies[0].sersic.centre[0] == 0.0
        assert instance.lens_galaxies[0].sis.centre[0] == 0.1
        assert instance.lens_galaxies[0].sis.centre[1] == 0.2
        assert instance.lens_galaxies[0].sis.einstein_radius == 0.3
        assert instance.lens_galaxies[0].redshift == 0.4
        assert instance.lens_galaxies[1].sis.centre[0] == 0.5
        assert instance.lens_galaxies[1].sis.centre[1] == 0.6
        assert instance.lens_galaxies[1].sis.einstein_radius == 0.7
        assert instance.lens_galaxies[1].redshift == 0.8

        class LensProfilePhase2(ph.LensProfilePhase):
            def pass_priors(self, previous_results):
                self.lens_galaxies[0].sis.einstein_radius = mm.Constant(10.0)

        phase = LensProfilePhase2(lens_galaxies=[gp.GalaxyPrior(sersic=light_profiles.EllipticalSersicLP,
                                                                sis=mass_profiles.SphericalIsothermalMP,
                                                                variable_redshift=True),
                                                 gp.GalaxyPrior(sis=mass_profiles.SphericalIsothermalMP,
                                                                variable_redshift=True)],
                                  optimizer_class=non_linear.MultiNest)

        # noinspection PyTypeChecker
        phase.pass_priors(None)

        instance = phase.optimizer.variable.instance_from_physical_vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2,
                                                                           0.4, 0.5, 0.6, 0.7, 0.8])
        instance += phase.optimizer.constant

        print(instance)

        assert instance.lens_galaxies[0].sersic.centre[0] == 0.0
        assert instance.lens_galaxies[0].sis.centre[0] == 0.1
        assert instance.lens_galaxies[0].sis.centre[1] == 0.2
        assert instance.lens_galaxies[0].sis.einstein_radius == 10.0
        assert instance.lens_galaxies[0].redshift == 0.4
        assert instance.lens_galaxies[1].sis.centre[0] == 0.5
        assert instance.lens_galaxies[1].sis.centre[1] == 0.6
        assert instance.lens_galaxies[1].sis.einstein_radius == 0.7
        assert instance.lens_galaxies[1].redshift == 0.8


# class TestPixelizedPhase(object):
#     def test_constructor(self):
#         phase = ph.PixelizedSourceLensAndPhase()
#         assert isinstance(phase.source_galaxies, gp.GalaxyPrior)
#         assert phase.lens_galaxies is None


# class TestAnalysis(object):
#     def test_model_image(self, results_collection, masked_image):
#         analysis = ph.LensProfilePhase.Analysis(results_collection, masked_image, "phase_name")
#         assert (results_collection[0].model_image == analysis.last_results.model_image).all()


class TestResult(object):

    # def test_hyper_galaxy_and_model_images(self):
    #
    #     analysis = MockAnalysis(number_galaxies=2, value=1.0)
    #
    # result = ph.LensMassAndSourceProfilePhase.Result(constant=mm.ModelInstance(), likelihood=1,
    # variable=mm.ModelMapper(), analysis=analysis) assert (result.image_plane_source_images[0] == np.array([
    # 1.0])).all() assert (result.image_plane_source_images[1] == np.array([1.0])).all() assert (
    # result.image == np.array([2.0])).all()

    def test_results(self):
        results = ph.ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1
