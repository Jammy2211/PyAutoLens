import os
from os import path

import numpy as np
import pytest

from autolens import conf
from autolens.autofit import model_mapper as mm
from autolens.autofit import non_linear
from autolens.galaxy import galaxy as g, galaxy_model as gm
from autolens.imaging import image as img
from autolens.imaging import mask as msk
from autolens.lensing import lensing_image as li
from autolens.pipeline import phase as ph
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result "
    "either in an error or a different result.")

directory = path.dirname(path.realpath(__file__))

general_conf = '{}/../test_files/configs/phase'.format(directory)
conf.instance.general = conf.NamedConfig("{}/general.ini".format(general_conf))

shape = (10, 10)


class MockAnalysis(object):

    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # def tracer_for_instance(self, instance):
    #     from autolens.lensing import ray_tracing
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
                instance += self.constant

                likelihood = analysis.fit(instance)
                self.result = non_linear.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.prior_count * [0.5])

        return fitness_function.result


@pytest.fixture(name="grids")
def make_grids(lensing_image):
    return msk.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(
        lensing_image.mask, 1, lensing_image.psf.shape)


@pytest.fixture(name="phase")
def make_phase():
    return ph.LensSourcePlanePhase(optimizer_class=NLO)


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_model")
def make_galaxy_model():
    return gm.GalaxyModel()


@pytest.fixture(name="image")
def make_image():
    image = img.Image(np.array(np.zeros(shape)), pixel_scale=1.0, psf=img.PSF(np.ones((3, 3)), pixel_scale=1.0),
                      noise_map=img.NoiseMap(np.ones(shape), pixel_scale=1.0))
    return image


@pytest.fixture(name="lensing_image")
def make_lensing_image():
    image = img.Image(np.array(np.zeros(shape)), pixel_scale=1.0, psf=img.PSF(np.ones((3, 3)), pixel_scale=1.0),
                      noise_map=img.NoiseMap(np.ones(shape), pixel_scale=1.0))
    mask = msk.Mask.circular(shape=shape, pixel_scale=1, radius_mask_arcsec=3.0)
    return li.LensingImage(image, mask)


@pytest.fixture(name="results")
def make_results():
    return MockResults(np.ones(shape),
                       galaxy_images=[np.ones(shape), np.ones(shape)])


@pytest.fixture(name="results_collection")
def make_results_collection(results):
    return ph.ResultsCollection([results])


class TestAutomaticPriorPassing(object):
    def test_galaxy_model_dict(self, phase, galaxy_model):
        phase.lens_galaxies = dict(galaxy_one=galaxy_model)
        assert phase.galaxy_model_tuples == [("galaxy_one", galaxy_model)]

    def test_match_galaxy_models_by_name(self, phase, galaxy_model, galaxy):
        phase.lens_galaxies = dict(galaxy_one=galaxy_model)
        instance = mm.ModelInstance()
        instance.galaxy_one = galaxy

        assert phase.match_instance_to_models(instance) == [("galaxy_one", galaxy, galaxy_model)]

    # noinspection PyUnresolvedReferences
    def test_update_galaxy_model_with_instance(self, phase):
        new_galaxy_model = gm.GalaxyModel(variable_redshift=True)
        mapper = mm.ModelMapper()

        mapper.galaxy_one = new_galaxy_model

        galaxy_model = gm.GalaxyModel(variable_redshift=True)

        phase.lens_galaxies = dict(galaxy_one=galaxy_model)

        phase.update_galaxy_models_with_mapper(mapper)

        assert phase.variable.galaxy_one == new_galaxy_model

        assert phase.lens_galaxies.galaxy_one == new_galaxy_model
        assert phase.lens_galaxies[0] == new_galaxy_model

        assert phase.lens_galaxies.galaxy_one != galaxy_model

    def test_phase_property_collections(self, phase):
        assert phase.phase_property_collections == [phase.lens_galaxies, phase.source_galaxies]

    # noinspection PyUnresolvedReferences
    def test_fit_priors(self, phase, galaxy_model, galaxy):
        argument_tuples = []

        new_galaxy_model = gm.GalaxyModel()

        def fitting_function(best_fit_galaxy, initial_galaxy_model):
            argument_tuples.append((best_fit_galaxy, initial_galaxy_model))
            return new_galaxy_model

        phase.lens_galaxies = dict(galaxy_one=galaxy_model)
        assert phase.lens_galaxies.galaxy_one is not None

        instance = mm.ModelInstance()
        instance.galaxy_one = galaxy

        phase.fit_priors(instance, fitting_function)

        assert phase.lens_galaxies.galaxy_one == new_galaxy_model
        assert argument_tuples == [(galaxy, galaxy_model)]

    def test_model_instance_sum_priority(self):
        instance_1 = mm.ModelInstance()
        galaxy_1 = g.Galaxy()
        instance_1.galaxy = galaxy_1

        instance_2 = mm.ModelInstance()
        galaxy_2 = g.Galaxy()
        instance_2.galaxy = galaxy_2

        assert (instance_1 + instance_2).galaxy == galaxy_2


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

    def test_set_variables(self, phase, galaxy_model):
        phase.lens_galaxies = [galaxy_model]
        assert phase.optimizer.variable.lens_galaxies == [galaxy_model]
        assert phase.optimizer.constant.lens_galaxies == []

    def test_make_analysis(self, phase, image, lensing_image):
        analysis = phase.make_analysis(image=image)
        assert analysis.last_results is None
        assert analysis.lensing_image == lensing_image

    def test_fit(self, image):
        clean_images()

        phase = ph.LensSourcePlanePhase(optimizer_class=NLO,
                                        lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                                        source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)])
        result = phase.run(image=image)
        assert isinstance(result.constant.lens_galaxies[0], g.Galaxy)
        assert isinstance(result.constant.source_galaxies[0], g.Galaxy)

    def test_customize(self, results, image):
        class MyPlanePhaseAnd(ph.LensSourcePlanePhase):
            def pass_priors(self, previous_results):
                self.lens_galaxies = previous_results.last.constant.lens_galaxies
                self.source_galaxies = previous_results.last.variable.source_galaxies

        galaxy = g.Galaxy()
        galaxy_model = gm.GalaxyModel()

        setattr(results.constant, "lens_galaxies", [galaxy])
        setattr(results.variable, "source_galaxies", [galaxy_model])

        phase = MyPlanePhaseAnd(optimizer_class=NLO)
        phase.make_analysis(image=image, previous_results=ph.ResultsCollection([results]))

        assert phase.lens_galaxies == [galaxy]
        assert phase.source_galaxies == [galaxy_model]

    def test_default_mask_function(self, phase, image):
        assert len(li.LensingImage(image, phase.mask_function(image))) == 32

    def test_duplication(self):
        phase = ph.LensSourcePlanePhase(lens_galaxies=[gm.GalaxyModel()], source_galaxies=[gm.GalaxyModel()])

        ph.LensSourcePlanePhase()

        assert phase.lens_galaxies is not None
        assert phase.source_galaxies is not None

    def test_modify_image(self, image):
        class MyPhase(ph.PhaseImaging):
            def modify_image(self, im, previous_results):
                assert image.shape == im.shape
                return im

        phase = MyPhase(phase_name='phase')
        analysis = phase.make_analysis(image)
        assert analysis.lensing_image != image

    def test_tracer_for_instance(self, image):
        lens_galaxy = g.Galaxy()
        source_galaxy = g.Galaxy()
        phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])
        analysis = phase.make_analysis(image)
        instance = phase.constant
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.source_plane.galaxies[0] == source_galaxy

    # TODO : Need to test using results

    # def test_unmasked_model_image_for_instance(self, _datas):
    #
    #     lens_galaxy = g.Galaxy(light_profile=lp.SphericalSersic(intensity=1.0))
    #     image_padded_grid = msk.PaddedImageGrid.unmasked_grid_from_shapes_and_pixel_scale(shape=_datas.shape,
    #                                                                                         psf_shape=_datas.psf.shape,
    #                                                                                         pixel_scale=_datas.pixel_scale)
    #     image_1d = lens_galaxy.intensities_from_grid(image_padded_grid)
    #     blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(image_1d, _datas.psf)
    #     blurred_image = image_padded_grid.scaled_array_from_array_1d(blurred_image_1d)
    #
    #     phase = ph.LensPlanePhase(lens_galaxies=[lens_galaxy])
    #     analysis = phase.make_analysis(_datas)
    #     instance = phase.constant
    #     unmasked_tracer = analysis.unmasked_tracer_for_instance(instance)
    #     unmasked_model_image = analysis.unmasked_model_image_for_tracer(unmasked_tracer)
    #
    #     assert blurred_image == pytest.approx(unmasked_model_image, 1e-4)
    #
    # def test_unmasked_model_images_of_galaxies_for_instance(self, _datas):
    #
    #     g0= g.Galaxy(light_profile=lp.SphericalSersic(intensity=1.0))
    #     g1 = g.Galaxy(light_profile=lp.SphericalSersic(intensity=2.0))
    #
    #     image_padded_grid = msk.PaddedImageGrid.unmasked_grid_from_shapes_and_pixel_scale(shape=_datas.shape,
    #                                                                                         psf_shape=_datas.psf.shape,
    #                                                                                         pixel_scale=_datas.pixel_scale)
    #
    #     g0_image_1d = g0.intensities_from_grid(image_padded_grid)
    #     g0_blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(g0_image_1d, _datas.psf)
    #     g0_blurred_image = image_padded_grid.scaled_array_from_array_1d(g0_blurred_image_1d)
    #
    #     g1_image_1d = g1.intensities_from_grid(image_padded_grid)
    #     g1_blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(g1_image_1d, _datas.psf)
    #     g1_blurred_image = image_padded_grid.scaled_array_from_array_1d(g1_blurred_image_1d)
    #
    #     phase = ph.LensPlanePhase(lens_galaxies=[g0, g1])
    #     analysis = phase.make_analysis(_datas)
    #     instance = phase.constant
    #     unmasked_tracer = analysis.unmasked_tracer_for_instance(instance)
    #     unmasked_model_images = analysis.unmasked_model_images_of_galaxies_for_tracer(unmasked_tracer)
    #
    #     assert g0_blurred_image == pytest.approx(unmasked_model_images[0], 1e-4)
    #     assert g1_blurred_image == pytest.approx(unmasked_model_images[1], 1e-4)

    def test__phase_can_receive_list_of_galaxy_models(self):
        phase = ph.LensPlanePhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic,
                                                                sis=mp.SphericalIsothermal,
                                                                variable_redshift=True),
                                                 gm.GalaxyModel(sis=mp.SphericalIsothermal,
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

        class LensPlanePhase2(ph.LensPlanePhase):
            # noinspection PyUnusedLocal
            def pass_models(self, previous_results):
                self.lens_galaxies[0].sis.einstein_radius = mm.Constant(10.0)

        phase = LensPlanePhase2(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic,
                                                              sis=mp.SphericalIsothermal,
                                                              variable_redshift=True),
                                               gm.GalaxyModel(sis=mp.SphericalIsothermal,
                                                              variable_redshift=True)],
                                optimizer_class=non_linear.MultiNest)

        # noinspection PyTypeChecker
        phase.pass_models(None)

        instance = phase.optimizer.variable.instance_from_physical_vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2,
                                                                           0.4, 0.5, 0.6, 0.7, 0.8])
        instance += phase.optimizer.constant

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
#         assert isinstance(phase.source_galaxies, gm.GalaxyModel)
#         assert phase.lens_galaxies is None


# class TestAnalysis(object):
#     def test_model_image(self, results_collection, lensing_image):
#         lensing = ph.LensPlanePhase.Analysis(results_collection, lensing_image, "analysis_path")
#         assert (results_collection[0].model_image == lensing.last_results.model_image).all()


class TestResult(object):

    # def test_hyper_galaxy_and_model_images(self):
    #
    #     lensing = MockAnalysis(number_galaxies=2, value=1.0)
    #
    # result = ph.LensSourcePlanePhase.Result(constant=mm.ModelInstance(), likelihood=1,
    # variable=mm.ModelMapper(), lensing=lensing) assert (result.image_plane_source_images[0] == np.array([
    # 1.0])).all() assert (result.image_plane_source_images[1] == np.array([1.0])).all() assert (
    # result._datas == np.array([2.0])).all()

    def test_results(self):
        results = ph.ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1
