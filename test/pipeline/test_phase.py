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
from autolens.profiles import light_profiles

shape = (10, 10)


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

                likelihood = analysis.fit(**instance.__dict__)
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
    return ph.SourceLensPhase(optimizer_class=NLO)


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


class TestPhase(object):

    def test_set_constants(self, phase, galaxy):
        phase.lens_galaxy = galaxy
        assert phase.optimizer.constant.lens_galaxy == galaxy
        assert not hasattr(phase.optimizer.variable, "lens_galaxy")

    def test_set_variables(self, phase, galaxy_prior):
        phase.lens_galaxy = galaxy_prior
        assert phase.optimizer.variable.lens_galaxy == galaxy_prior
        assert not hasattr(phase.optimizer.constant, "lens_galaxy")

    def test_mask_analysis(self, phase, image, masked_image):
        analysis = phase.make_analysis(image=image)
        assert analysis.last_results is None
        assert analysis.masked_image == masked_image

    def test_fit(self, phase, image):
        phase.source_galaxy = g.Galaxy()
        phase.lens_galaxy = g.Galaxy()
        result = phase.run(image=image)
        assert isinstance(result.constant.lens_galaxy, g.Galaxy)
        assert isinstance(result.constant.source_galaxy, g.Galaxy)

    def test_customize(self, results, image):
        class MyPhase(ph.SourceLensPhase):
            def pass_priors(self, previous_results):
                self.lens_galaxy = previous_results.last.constant.lens_galaxy
                self.source_galaxy = previous_results.last.variable.source_galaxy

        galaxy = g.Galaxy()
        galaxy_prior = gp.GalaxyPrior()

        setattr(results.constant, "lens_galaxy", galaxy)
        setattr(results.variable, "source_galaxy", galaxy_prior)

        phase = MyPhase(optimizer_class=NLO)
        phase.make_analysis(image=image, previous_results=ph.ResultsCollection([results]))

        assert phase.lens_galaxy == galaxy
        assert phase.source_galaxy == galaxy_prior

    def test_phase_property(self):

        class MyPhase(ph.SourceLensPhase):
            prop = ph.phase_property("prop")

        phase = MyPhase(optimizer_class=NLO)

        phase.prop = gp.GalaxyPrior()

        assert phase.variable.prop == phase.prop

        galaxy = g.Galaxy()
        phase.prop = galaxy

        assert phase.constant.prop == galaxy
        assert not hasattr(phase.variable, "prop")

        phase.prop = gp.GalaxyPrior()
        assert not hasattr(phase.constant, "prop")

    def test_default_mask_function(self, phase, image):
        assert len(mi.MaskedImage(image, phase.mask_function(image))) == 32

    def test_galaxy_images(self, image, phase):
        phase.lens_galaxy = g.Galaxy()
        phase.source_galaxy = g.Galaxy()
        result = phase.run(image)
        assert len(result.galaxy_images) == 2

    def test_duplication(self):
        phase = ph.SourceLensPhase(lens_galaxy=gp.GalaxyPrior(), source_galaxy=gp.GalaxyPrior())

        ph.SourceLensPhase()

        assert phase.lens_galaxy is not None
        assert phase.source_galaxy is not None

    def test_modify_image(self, image):
        class MyPhase(ph.Phase):
            def modify_image(self, im, previous_results):
                assert image.shape == im.shape
                return im

        phase = MyPhase()
        analysis = phase.make_analysis(image)
        assert analysis.masked_image != image

    def test_model_images(self, image):
        phase = ph.SourceLensPhase()
        analysis = phase.make_analysis(image)
        instance = mm.ModelInstance()
        instance.lens_galaxy = g.Galaxy(light=light_profiles.EllipticalExponential())
        instance.source_galaxy = None

        images = analysis.galaxy_images_for_model(instance)
        assert images[0].shape == image.shape
        assert images[1] is None


class TestPixelizedPhase(object):
    def test_constructor(self):
        phase = ph.PixelizedSourceLensPhase()
        assert isinstance(phase.source_galaxy, gp.GalaxyPrior)
        assert phase.lens_galaxy is None


class TestAnalysis(object):
    def test_model_image(self, results_collection, masked_image):
        analysis = ph.Phase.Analysis(results_collection, masked_image)
        assert (results_collection[0].model_image == analysis.last_results.model_image).all()


class TestResult(object):
    def test_model_image(self):
        result = ph.Phase.Result(mm.ModelInstance(), 1, mm.ModelMapper(), [np.array([1, 2, 3]), np.array([2, 3, 4])])
        assert (result.model_image == np.array([3, 5, 7])).all()

    def test_results(self):
        results = ph.ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1
