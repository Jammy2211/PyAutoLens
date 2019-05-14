import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit.tools.pipeline
from autofit import conf
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior
from autofit.optimize import non_linear
from autolens import exc
from autolens.data import ccd
from autolens.data.array import grids, mask as msk
from autolens.data.array import scaled_array
from autolens.lens import lens_data as ld
from autolens.lens import lens_fit
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline import phase as ph

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result "
    "either in an error or a different result.")

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config('{}/../test_files/configs/phase'.format(directory))


shape = (10, 10)


class MockAnalysis(object):

    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]


class MockResults(object):
    def __init__(self, model_image=None, galaxy_images=(), constant=None, analysis=None, optimizer=None):
        self.model_image = model_image
        self.galaxy_images = galaxy_images
        self.constant = constant or mm.ModelInstance()
        self.variable = mm.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer


class MockResult:
    def __init__(self, constant, figure_of_merit, variable=None):
        self.constant = constant
        self.figure_of_merit = figure_of_merit
        self.variable = variable
        self.previous_variable = variable
        self.gaussian_tuples = None


class NLO(non_linear.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector)
        fitness_function(self.variable.prior_count * [0.8])

        return fitness_function.result


@pytest.fixture(name="grid_stack")
def make_grids(lens_data):
    return grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
        lens_data.mask, 1, lens_data.psf.shape)


@pytest.fixture(name="phase")
def make_phase():
    return ph.LensSourcePlanePhase(optimizer_class=NLO, mask_function=ph.default_mask_function, phase_name='test_phase')


@pytest.fixture(name="galaxy")
def make_galaxy():
    return g.Galaxy()


@pytest.fixture(name="galaxy_model")
def make_galaxy_model():
    return gm.GalaxyModel()


@pytest.fixture(name="ccd_data")
def make_ccd_data():
    pixel_scale = 1.0

    image = scaled_array.ScaledSquarePixelArray(array=np.array(np.zeros(shape)), pixel_scale=pixel_scale)
    psf = ccd.PSF(array=np.ones((3, 3)), pixel_scale=pixel_scale)
    noise_map = ccd.NoiseMap(np.ones(shape), pixel_scale=1.0)

    return ccd.CCDData(image=image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map)


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=shape, pixel_scale=1, radius_arcsec=3.0)


@pytest.fixture(name="lens_data")
def make_lens_image(mask):
    ccd_data = ccd.CCDData(np.array(np.zeros(shape)), pixel_scale=1.0, psf=ccd.PSF(np.ones((3, 3)), pixel_scale=1.0),
                           noise_map=ccd.NoiseMap(np.ones(shape), pixel_scale=1.0))
    mask = msk.Mask.circular(shape=shape, pixel_scale=1, radius_arcsec=3.0)
    return ld.LensData(ccd_data=ccd_data, mask=mask)


@pytest.fixture(name="results")
def make_results():
    return MockResults(np.ones(shape),
                       galaxy_images=[np.ones(shape), np.ones(shape)])


@pytest.fixture(name="results_collection")
def make_results_collection(results):
    results_collection = autofit.tools.pipeline.ResultsCollection()
    results_collection.add("phase", results)
    return results_collection


class MockLensData(object):
    def __init__(self, ccd_data, noise_map, mask):
        self.ccd_data = ccd_data
        self.image = ccd_data
        self.noise_map = noise_map
        self.mask = mask


@pytest.fixture(name="hyper_lens_data")
def make_lens_data():
    return MockLensData(np.ones(5), np.ones(5), np.full(5, True))


@pytest.fixture(name="hyper_galaxy")
def make_hyper_galaxy():
    return g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)


@pytest.fixture(name="hyper_phase")
def make_hyper_phase():
    return ph.HyperGalaxyPhase("hyper_galaxy_phase")


class TestRedshift(object):
    def test_lens_phase(self):
        phase = ph.LensPlanePhase("lens phase")
        phase.lens_galaxies = [g.Galaxy(), gm.GalaxyModel()]

        assert phase.lens_galaxies[0].redshift == 0.5
        assert phase.lens_galaxies[1].redshift == 0.5

    def test_lens_source_phase(self):
        phase = ph.LensSourcePlanePhase("lens source phase")
        phase.lens_galaxies = [g.Galaxy(), gm.GalaxyModel()]
        phase.source_galaxies = [g.Galaxy(), gm.GalaxyModel()]

        assert phase.lens_galaxies[0].redshift == 0.5
        assert phase.lens_galaxies[1].redshift == 0.5

        assert phase.source_galaxies[0].redshift == 1.0
        assert phase.source_galaxies[1].redshift == 1.0

    def test_truthy_not_overridden(self):
        phase = ph.LensSourcePlanePhase("lens source phase")
        phase.lens_galaxies = [g.Galaxy(redshift=0.1), gm.GalaxyModel()]
        phase.source_galaxies = [g.Galaxy(), gm.GalaxyModel(redshift=2.0)]

        assert phase.lens_galaxies[0].redshift == 0.1
        assert phase.lens_galaxies[1].redshift == 0.5

        assert phase.source_galaxies[0].redshift == 1.0
        assert phase.source_galaxies[1].redshift == 2.0


class TestHyperGalaxyPhase(object):
    def test_analysis(self, hyper_lens_data, hyper_galaxy):
        analysis = ph.HyperGalaxyPhase.Analysis(hyper_lens_data, np.ones(5), np.ones(5))
        result = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)

        assert isinstance(result, lens_fit.LensDataFit)

    def test_run(self, hyper_galaxy, hyper_phase, hyper_lens_data):
        class Instance(object):
            def __init__(self):
                self.hyper_galaxy = "hyper_galaxy"
                self.one = g.Galaxy()
                self.two = g.Galaxy()
                self.three = g.Galaxy()

            @staticmethod
            def name_instance_tuples_for_class(cls):
                return [("one", None), ("two", None), ("three", None)]

        class MockOptimizer(object):
            def __init__(self):
                self.extensions = []
                self.constant = Instance()
                self.variable = Instance()

            @classmethod
            def fit(cls, analysis):
                instance = mm.ModelInstance()
                instance.hyper_galaxy = hyper_galaxy
                return analysis.fit(instance)

            def copy_with_name_extension(self, name):
                self.extensions.append(name)
                return self

        optimizer = MockOptimizer()
        hyper_phase.optimizer = optimizer

        class Result(object):
            def __init__(self):
                self.constant \
                    = Instance()
                self.variable = Instance()

            def unmasked_image_for_galaxy(self, galaxy):
                return np.ones(5)

            @property
            def unmasked_model_image(self):
                return np.ones(5)

        class PreviousResults(object):
            @property
            def last(self):
                return Result()

        results = hyper_phase.run(hyper_lens_data, PreviousResults())

        assert isinstance(results, Result)
        assert optimizer.extensions == ["one", "two", "three"]
        assert results.variable.one.hyper_galaxy == g.HyperGalaxy
        assert results.constant.one.hyper_galaxy == "hyper_galaxy"

    def test__figure_of_merit_of_fit__noise_factor_0_so_no_noise_scaling(self, hyper_phase):
        hyper_lens_data = MockLensData(ccd_data=np.ones(3), noise_map=np.ones(3), mask=np.full(3, False))
        analysis = ph.HyperGalaxyPhase.Analysis(lens_data=hyper_lens_data, model_image=np.ones(3),
                                                galaxy_image=np.ones(3))
        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)
        fit = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)
        assert (fit.residual_map == np.zeros(3)).all()
        assert (fit.chi_squared_map == np.zeros(3)).all()
        assert (fit.noise_map == hyper_lens_data.noise_map).all()

        chi_squared = 0.0
        noise_normalization = 3.0 * np.log(2 * np.pi * 1.0 ** 2.0)

        assert fit.figure_of_merit == -0.5 * (chi_squared + noise_normalization)

        hyper_lens_data = MockLensData(ccd_data=np.ones(3), noise_map=2.0 * np.ones(3), mask=np.full(3, False))
        analysis = ph.HyperGalaxyPhase.Analysis(lens_data=hyper_lens_data, model_image=np.ones(3),
                                                galaxy_image=np.ones(3))
        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)
        fit = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)
        assert (fit.residual_map == np.zeros(3)).all()
        assert (fit.chi_squared_map == np.zeros(3)).all()
        assert (fit.noise_map == hyper_lens_data.noise_map).all()

        chi_squared = 0.0
        noise_normalization = 3.0 * np.log(2 * np.pi * 2.0 ** 2.0)

        assert fit.figure_of_merit == -0.5 * (chi_squared + noise_normalization)

        hyper_lens_data = MockLensData(ccd_data=2.0 * np.ones(3), noise_map=2.0 * np.ones(3), mask=np.full(3, False))
        analysis = ph.HyperGalaxyPhase.Analysis(lens_data=hyper_lens_data, model_image=np.ones(3),
                                                galaxy_image=np.ones(3))
        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)
        fit = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)
        assert (fit.residual_map == np.ones(3)).all()
        assert (fit.chi_squared_map == 0.25 * np.ones(3)).all()
        assert (fit.noise_map == hyper_lens_data.noise_map).all()

        chi_squared = 0.75
        noise_normalization = 3.0 * np.log(2 * np.pi * 2.0 ** 2.0)

        assert fit.figure_of_merit == -0.5 * (chi_squared + noise_normalization)

    def test__figure_of_merit_of_fit__hyper_galaxy_params_scale_noise_as_expected(self, hyper_phase):
        hyper_lens_data = MockLensData(ccd_data=np.ones(3), noise_map=np.ones(3), mask=np.full(3, False))
        analysis = ph.HyperGalaxyPhase.Analysis(lens_data=hyper_lens_data, model_image=np.ones(3),
                                                galaxy_image=np.ones(3))
        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)
        fit = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)
        assert (fit.residual_map == np.zeros(3)).all()
        assert (fit.chi_squared_map == np.zeros(3)).all()
        assert (fit.noise_map == hyper_lens_data.noise_map + np.ones(3)).all()

        chi_squared = 0.0
        noise_normalization = 3.0 * np.log(2 * np.pi * 2.0 ** 2.0)

        assert fit.figure_of_merit == -0.5 * (chi_squared + noise_normalization)

        hyper_lens_data = MockLensData(ccd_data=np.ones(3), noise_map=2.0 * np.ones(3), mask=np.full(3, False))
        analysis = ph.HyperGalaxyPhase.Analysis(lens_data=hyper_lens_data, model_image=np.ones(3),
                                                galaxy_image=2.0 * np.ones(3))
        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0)
        fit = analysis.fit_for_hyper_galaxy(hyper_galaxy=hyper_galaxy)
        assert (fit.residual_map == np.zeros(3)).all()
        assert (fit.chi_squared_map == np.zeros(3)).all()
        assert (fit.noise_map == hyper_lens_data.noise_map + ((2.0 * np.ones(3)) ** 2.0)).all()

        chi_squared = 0.0
        noise_normalization = 3.0 * np.log(2 * np.pi * 6.0 ** 2.0)

        assert fit.figure_of_merit == -0.5 * (chi_squared + noise_normalization)


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
        assert phase.optimizer.variable.lens_galaxies == [galaxy]

    def test_set_variables(self, phase, galaxy_model):
        phase.lens_galaxies = [galaxy_model]
        assert phase.optimizer.variable.lens_galaxies == [galaxy_model]

    def test_make_analysis(self, phase, ccd_data, lens_data):
        analysis = phase.make_analysis(data=ccd_data)
        assert analysis.last_results is None
        assert analysis.lens_data.image == ccd_data.image
        assert analysis.lens_data.noise_map == ccd_data.noise_map
        assert analysis.lens_data.image == lens_data.image
        assert analysis.lens_data.noise_map == lens_data.noise_map

    def test_make_analysis__mask_input_uses_mask__no_mask_uses_mask_function(self, phase, ccd_data):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase.mask_function = None

        mask_input = msk.Mask.circular(shape=shape, pixel_scale=1, radius_arcsec=2.0)

        analysis = phase.make_analysis(data=ccd_data, mask=mask_input)
        assert (analysis.lens_data.mask == mask_input).all()

        # If a mask function is suppled, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return msk.Mask.circular(shape=image.shape, pixel_scale=1, radius_arcsec=1.4)

        mask_from_function = mask_function(image=ccd_data.image)
        phase.mask_function = mask_function

        analysis = phase.make_analysis(data=ccd_data, mask=None)
        assert (analysis.lens_data.mask == mask_from_function).all()
        analysis = phase.make_analysis(data=ccd_data, mask=mask_input)
        assert (analysis.lens_data.mask == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask.

        mask_default = ph.default_mask_function(image=ccd_data.image)
        phase.mask_function = None
        analysis = phase.make_analysis(data=ccd_data, mask=None)
        assert (analysis.lens_data.mask == mask_default).all()

    def test_make_analysis__mask_input_uses_mask__inner_mask_radius_included_which_masks_centre(self, phase, ccd_data):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase.mask_function = None
        phase.inner_mask_radii = 1.0

        mask_input = msk.Mask.circular(shape=shape, pixel_scale=1, radius_arcsec=2.0)

        analysis = phase.make_analysis(data=ccd_data, mask=mask_input)

        # The inner circulaar mask radii of 1.0" masks the centra pixels of the mask
        mask_input[4:6, 4:6] = True

        assert (analysis.lens_data.mask == mask_input).all()

        # If a mask function is suppled, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return msk.Mask.circular(shape=image.shape, pixel_scale=1, radius_arcsec=1.4)

        mask_from_function = mask_function(image=ccd_data.image)
        # The inner circulaar mask radii of 1.0" masks the centra pixels of the mask
        mask_from_function[4:6, 4:6] = True

        phase.mask_function = mask_function

        analysis = phase.make_analysis(data=ccd_data, mask=None)
        assert (analysis.lens_data.mask == mask_from_function).all()
        analysis = phase.make_analysis(data=ccd_data, mask=mask_input)
        assert (analysis.lens_data.mask == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask.

        mask_default = ph.default_mask_function(image=ccd_data.image)
        # The inner circulaar mask radii of 1.0" masks the centra pixels of the mask
        mask_default[4:6, 4:6] = True

        phase.mask_function = None
        analysis = phase.make_analysis(data=ccd_data, mask=None)
        assert (analysis.lens_data.mask == mask_default).all()

    def test_make_analysis__positions_are_input__are_used_in_analysis(self, phase, ccd_data):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens data.

        phase.positions_threshold = 0.2

        analysis = phase.make_analysis(data=ccd_data, positions=[[[1.0, 1.0], [2.0, 2.0]]])
        assert (analysis.lens_data.positions[0][0] == np.array([1.0, 1.0])).all()
        assert (analysis.lens_data.positions[0][1] == np.array([2.0, 2.0])).all()

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):
            phase.make_analysis(data=ccd_data, positions=None)
            phase.make_analysis(data=ccd_data)

        # If positions threshold is None, positions should always be None.

        phase.positions_threshold = None
        analysis = phase.make_analysis(data=ccd_data, positions=[[[1.0, 1.0], [2.0, 2.0]]])
        assert analysis.lens_data.positions is None

    def test_make_analysis__interp_pixel_scale_is_input__interp_grid_used_in_analysis(self, phase, ccd_data):
        # If use positions is true and positions are input, make the positions part of the lens data.

        phase.interp_pixel_scale = 0.1

        analysis = phase.make_analysis(data=ccd_data)
        assert analysis.lens_data.interp_pixel_scale == 0.1
        assert hasattr(analysis.lens_data.grid_stack.regular, 'interpolator')
        assert hasattr(analysis.lens_data.grid_stack.sub, 'interpolator')
        assert hasattr(analysis.lens_data.grid_stack.blurring, 'interpolator')
        assert hasattr(analysis.lens_data.padded_grid_stack.regular, 'interpolator')
        assert hasattr(analysis.lens_data.padded_grid_stack.sub, 'interpolator')

    def test__make_analysis__phase_info_is_made(self, phase, ccd_data):
        phase.make_analysis(data=ccd_data)

        file_phase_info = "{}/{}".format(phase.optimizer.phase_output_path, 'phase.info')

        phase_info = open(file_phase_info, 'r')

        optimizer = phase_info.readline()
        sub_grid_size = phase_info.readline()
        image_psf_shape = phase_info.readline()
        pixelization_psf_shape = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()
        auto_link_priors = phase_info.readline()

        phase_info.close()

        assert optimizer == 'Optimizer = NLO \n'
        assert sub_grid_size == 'Sub-grid size = 2 \n'
        assert image_psf_shape == 'Image PSF shape = None \n'
        assert pixelization_psf_shape == 'Pixelization PSF shape = None \n'
        assert positions_threshold == 'Positions Threshold = None \n'
        assert cosmology == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, ' \
                            'Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n'
        assert auto_link_priors == 'Auto Link Priors = False \n'

    def test_fit(self, ccd_data):
        clean_images()

        phase = ph.LensSourcePlanePhase(optimizer_class=NLO,
                                        lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                        source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                        phase_name='test_phase')
        result = phase.run(data=ccd_data)
        assert isinstance(result.constant.lens_galaxies[0], g.Galaxy)
        assert isinstance(result.constant.source_galaxies[0], g.Galaxy)

    def test_customize(self, results, results_collection, ccd_data):
        class MyPlanePhaseAnd(ph.LensSourcePlanePhase):
            def pass_priors(self, results):
                self.lens_galaxies = results.last.constant.lens_galaxies
                self.source_galaxies = results.last.variable.source_galaxies

        galaxy = g.Galaxy()
        galaxy_model = gm.GalaxyModel()

        setattr(results.constant, "lens_galaxies", [galaxy])
        setattr(results.variable, "source_galaxies", [galaxy_model])

        phase = MyPlanePhaseAnd(optimizer_class=NLO, phase_name='test_phase')
        phase.make_analysis(data=ccd_data, results=results_collection)

        assert phase.lens_galaxies == [galaxy]
        assert phase.source_galaxies == [galaxy_model]

    def test_default_mask_function(self, phase, ccd_data):
        lens_data = ld.LensData(ccd_data=ccd_data, mask=phase.mask_function(ccd_data.image))
        assert len(lens_data.image_1d) == 32

    def test_duplication(self):
        phase = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel()),
                                        source_galaxies=dict(source=gm.GalaxyModel()),
                                        phase_name='test_phase')

        ph.LensSourcePlanePhase(phase_name='test_phase')

        assert phase.lens_galaxies is not None
        assert phase.source_galaxies is not None

    def test_modify_image(self, ccd_data):
        class MyPhase(ph.PhaseImaging):
            def modify_image(self, image, results):
                assert ccd_data.image.shape == image.shape
                image = 20.0 * np.ones(shape=shape)
                return image

        phase = MyPhase(phase_name='phase')
        analysis = phase.make_analysis(data=ccd_data)
        assert (analysis.lens_data.image == 20.0 * np.ones(shape=shape)).all()
        assert (analysis.lens_data.image_1d == 20.0 * np.ones(shape=32)).all()

    def test__check_if_phase_uses_inversion(self):

        phase = ph.LensPlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)))

        assert phase.uses_inversion == False

        phase = ph.LensPlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, pixelization=pix.Rectangular,
                                                   regularization=reg.Constant)))

        assert phase.uses_inversion == True

        phase = ph.LensSourcePlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)),
            source_galaxies=dict(source=gm.GalaxyModel(redshift=1.0)))

        assert phase.uses_inversion == False

        phase = ph.LensSourcePlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, pixelization=pix.Rectangular,
                                                   regularization=reg.Constant)),
            source_galaxies=dict(source=gm.GalaxyModel(redshift=1.0)))

        assert phase.uses_inversion == True

        phase = ph.LensSourcePlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)),
            source_galaxies=dict(source=gm.GalaxyModel(redshift=1.0, pixelization=pix.Rectangular,
                                                       regularization=reg.Constant)))

        assert phase.uses_inversion == True

    def test__phase_with_no_inversion__convolver_mapping_matrix_of_lens_data_is_none(self, ccd_data, mask):

        phase = ph.LensPlanePhase(
            phase_name='test_phase',
            lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)))

        analysis = phase.make_analysis(data=ccd_data)

        assert analysis.lens_data.convolver_mapping_matrix is None

    def test__lens_data_is_binned_up(self, ccd_data, mask):

        binned_up_ccd_data = ccd_data.new_ccd_data_with_binned_up_arrays(bin_up_factor=2)
        binned_up_mask = mask.binned_up_mask_from_mask(bin_up_factor=2)

        phase = ph.PhaseImaging(phase_name='phase', bin_up_factor=2)
        analysis = phase.make_analysis(data=ccd_data)
        assert (analysis.lens_data.image == binned_up_ccd_data.image).all()
        assert (analysis.lens_data.psf == binned_up_ccd_data.psf).all()
        assert (analysis.lens_data.noise_map == binned_up_ccd_data.noise_map).all()

        assert (analysis.lens_data.mask == binned_up_mask).all()

        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
        binned_up_lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(bin_up_factor=2)

        assert (analysis.lens_data.image == binned_up_lens_data.image).all()
        assert (analysis.lens_data.psf == binned_up_lens_data.psf).all()
        assert (analysis.lens_data.noise_map == binned_up_lens_data.noise_map).all()

        assert (analysis.lens_data.mask == binned_up_lens_data.mask).all()

        assert (analysis.lens_data.image_1d == binned_up_lens_data.image_1d).all()
        assert (analysis.lens_data.noise_map_1d == binned_up_lens_data.noise_map_1d).all()

    def test__tracer_for_instance__includes_cosmology(self, ccd_data):

        lens_galaxy = g.Galaxy()
        source_galaxy = g.Galaxy()

        phase = ph.LensPlanePhase(lens_galaxies=[lens_galaxy], cosmology=cosmo.FLRW, phase_name='test_phase')
        analysis = phase.make_analysis(ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)
        padded_tracer = analysis.padded_tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.cosmology == cosmo.FLRW
        assert padded_tracer.image_plane.galaxies[0] == lens_galaxy
        assert padded_tracer.cosmology == cosmo.FLRW

        phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                        cosmology=cosmo.FLRW, phase_name='test_phase')
        analysis = phase.make_analysis(ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)
        padded_tracer = analysis.padded_tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.source_plane.galaxies[0] == source_galaxy
        assert tracer.cosmology == cosmo.FLRW
        assert padded_tracer.image_plane.galaxies[0] == lens_galaxy
        assert padded_tracer.source_plane.galaxies[0] == source_galaxy
        assert padded_tracer.cosmology == cosmo.FLRW

        galaxy_0 = g.Galaxy(redshift=0.1)
        galaxy_1 = g.Galaxy(redshift=0.2)
        galaxy_2 = g.Galaxy(redshift=0.3)

        phase = ph.MultiPlanePhase(galaxies=[galaxy_0, galaxy_1, galaxy_2], cosmology=cosmo.WMAP7,
                                   phase_name='test_phase')
        analysis = phase.make_analysis(data=ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)
        padded_tracer = analysis.padded_tracer_for_instance(instance)

        assert tracer.planes[0].galaxies[0] == galaxy_0
        assert tracer.planes[1].galaxies[0] == galaxy_1
        assert tracer.planes[2].galaxies[0] == galaxy_2
        assert tracer.cosmology == cosmo.WMAP7
        assert padded_tracer.planes[0].galaxies[0] == galaxy_0
        assert padded_tracer.planes[1].galaxies[0] == galaxy_1
        assert padded_tracer.planes[2].galaxies[0] == galaxy_2
        assert padded_tracer.cosmology == cosmo.WMAP7

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(self, ccd_data):

        lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=0.1))
        source_galaxy = g.Galaxy(pixelization=pix.Rectangular(shape=(4, 4)),
                                 regularization=reg.Constant(coefficients=(1.0,)))

        phase = ph.LensPlanePhase(lens_galaxies=[lens_galaxy], mask_function=ph.default_mask_function,
                                  cosmology=cosmo.FLRW, phase_name='test_phase')
        analysis = phase.make_analysis(data=ccd_data)
        instance = phase.variable.instance_from_unit_vector([])

        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(image=ccd_data.image)
        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

        phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                        mask_function=ph.default_mask_function, cosmology=cosmo.FLRW,
                                        phase_name='test_phase')
        phase.uses_inversion = True
        analysis = phase.make_analysis(data=ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(image=ccd_data.image)
        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileInversionFit(lens_data=lens_data, tracer=tracer)

        assert fit.evidence == fit_figure_of_merit

    # TODO : Need to test using results

    # def test_unmasked_model_image_for_instance(self, image_):
    #
    #     lens_galaxy = g.Galaxy(light_profile=lp.SphericalSersic(intensity=1.0))
    #     image_padded_grid = msk.PaddedRegularGrid.unmasked_grid_from_shapes_and_pixel_scale(shape=image_.shape,
    #                                                                                         psf_shape=image_.psf.shape,
    #                                                                                         pixel_scale=image_.pixel_scale)
    #     image_1d = lens_galaxy.intensities_from_grid(image_padded_grid)
    #     blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(image_1d, image_.psf)
    #     blurred_image = image_padded_grid.scaled_array_from_array_1d(blurred_image_1d)
    #
    #     phase = ph.LensPlanePhase(lens_galaxies=[lens_galaxy])
    #     analysis = phase.make_analysis(image_)
    #     instance = phase.constant
    #     unmasked_tracer = analysis.unmasked_tracer_for_instance(instance)
    #     unmasked_model_image = analysis.unmasked_model_image_for_tracer(unmasked_tracer)
    #
    #     assert blurred_image == pytest.approx(unmasked_model_image, 1e-4)
    #
    # def test_unmasked_model_images_of_galaxies_for_instance(self, image_):
    #
    #     g0= g.Galaxy(light_profile=lp.SphericalSersic(intensity=1.0))
    #     g1 = g.Galaxy(light_profile=lp.SphericalSersic(intensity=2.0))
    #
    #     image_padded_grid = msk.PaddedRegularGrid.unmasked_grid_from_shapes_and_pixel_scale(shape=image_.shape,
    #                                                                                         psf_shape=image_.psf.shape,
    #                                                                                         pixel_scale=image_.pixel_scale)
    #
    #     g0_image_1d = g0.intensities_from_grid(image_padded_grid)
    #     g0_blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(g0_image_1d, image_.psf)
    #     g0_blurred_image = image_padded_grid.scaled_array_from_array_1d(g0_blurred_image_1d)
    #
    #     g1_image_1d = g1.intensities_from_grid(image_padded_grid)
    #     g1_blurred_image_1d = image_padded_grid.convolve_array_1d_with_psf(g1_image_1d, image_.psf)
    #     g1_blurred_image = image_padded_grid.scaled_array_from_array_1d(g1_blurred_image_1d)
    #
    #     phase = ph.LensPlanePhase(lens_galaxies=[g0, g1])
    #     analysis = phase.make_analysis(image_)
    #     instance = phase.constant
    #     unmasked_tracer = analysis.unmasked_tracer_for_instance(instance)
    #     unmasked_model_images = analysis.unmasked_model_images_of_galaxies_for_tracer(unmasked_tracer)
    #
    #     assert g0_blurred_image == pytest.approx(unmasked_model_images[0], 1e-4)
    #     assert g1_blurred_image == pytest.approx(unmasked_model_images[1], 1e-4)

    def test__phase_can_receive_list_of_galaxy_models(self):
        phase = ph.LensPlanePhase(
            lens_galaxies=dict(lens=gm.GalaxyModel(sersic=lp.EllipticalSersic, sis=mp.SphericalIsothermal,
                                                   redshift=g.Redshift),
                               lens1=gm.GalaxyModel(sis=mp.SphericalIsothermal, redshift=g.Redshift)),
            optimizer_class=non_linear.MultiNest, phase_name='test_phase')

        instance = phase.optimizer.variable.instance_from_physical_vector(
            [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.8, 0.1, 0.2, 0.3,
             0.4, 0.9, 0.5, 0.7, 0.8])

        assert instance.lens_galaxies[0].sersic.centre[0] == 0.2
        assert instance.lens_galaxies[0].sis.centre[0] == 0.1
        assert instance.lens_galaxies[0].sis.centre[1] == 0.2
        assert instance.lens_galaxies[0].sis.einstein_radius == 0.3
        assert instance.lens_galaxies[0].redshift == 0.4
        assert instance.lens_galaxies[1].sis.centre[0] == 0.9
        assert instance.lens_galaxies[1].sis.centre[1] == 0.5
        assert instance.lens_galaxies[1].sis.einstein_radius == 0.7
        assert instance.lens_galaxies[1].redshift == 0.8

        class LensPlanePhase2(ph.LensPlanePhase):
            # noinspection PyUnusedLocal
            def pass_models(self, results):
                self.lens_galaxies[0].sis.einstein_radius = prior.Constant(10.0)

        phase = LensPlanePhase2(
            lens_galaxies=dict(lens=gm.GalaxyModel(sersic=lp.EllipticalSersic, sis=mp.SphericalIsothermal,
                                                 redshift=g.Redshift),
                                lens1=gm.GalaxyModel(sis=mp.SphericalIsothermal, redshift=g.Redshift)),
                                optimizer_class=non_linear.MultiNest, phase_name='test_phase')

        # noinspection PyTypeChecker
        phase.pass_models(None)

        instance = phase.optimizer.variable.instance_from_physical_vector(
            [0.01, 0.02, 0.23, 0.04, 0.05, 0.06, 0.87, 0.1, 0.2,
             0.4, 0.5, 0.5, 0.7, 0.8])

        assert instance.lens_galaxies[0].sersic.centre[0] == 0.01
        assert instance.lens_galaxies[0].sis.centre[0] == 0.1
        assert instance.lens_galaxies[0].sis.centre[1] == 0.2
        assert instance.lens_galaxies[0].sis.einstein_radius == 10.0
        assert instance.lens_galaxies[0].redshift == 0.4
        assert instance.lens_galaxies[1].sis.centre[0] == 0.5
        assert instance.lens_galaxies[1].sis.centre[1] == 0.5
        assert instance.lens_galaxies[1].sis.einstein_radius == 0.7
        assert instance.lens_galaxies[1].redshift == 0.8


class TestResult(object):
    def test__results_of_phase_are_available_as_properties(self, ccd_data):
        clean_images()

        phase = ph.LensPlanePhase(optimizer_class=NLO,
                                  lens_galaxies=[g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))],
                                  phase_name='test_phase')

        result = phase.run(data=ccd_data)

        assert isinstance(result, ph.AbstractPhase.Result)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(self, ccd_data):
        lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=0.1))
        source_galaxy = g.Galaxy(pixelization=pix.Rectangular(shape=(4, 4)),
                                 regularization=reg.Constant(coefficients=(1.0,)))

        phase = ph.LensPlanePhase(lens_galaxies=[lens_galaxy], mask_function=ph.default_mask_function,
                                  cosmology=cosmo.FLRW, phase_name='test_phase')
        analysis = phase.make_analysis(data=ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(image=ccd_data.image)
        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

        phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                        mask_function=ph.default_mask_function, cosmology=cosmo.FLRW,
                                        phase_name='test_phase')

        phase.uses_inversion = True

        analysis = phase.make_analysis(data=ccd_data)
        instance = phase.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(image=ccd_data.image)
        lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileInversionFit(lens_data=lens_data, tracer=tracer)

        assert fit.evidence == fit_figure_of_merit
