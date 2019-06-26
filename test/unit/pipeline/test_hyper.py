import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.lens import lens_fit
from autolens.lens import ray_tracing as rt
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline.phase import phase_hyper
from autolens.pipeline.phase import phase_imaging
from test.unit.mock.pipeline import mock_pipeline


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return g.Galaxy(redshift=1.0, light=lp.SphericalSersic(),
                    mass=mp.SphericalIsothermal())


@pytest.fixture(name="source_galaxy")
def make_source_galaxy():
    return g.Galaxy(redshift=2.0, light=lp.SphericalSersic())


@pytest.fixture(name="lens_galaxies")
def make_lens_galaxies(lens_galaxy):
    lens_galaxies = af.ModelInstance()
    lens_galaxies.lens = lens_galaxy
    return lens_galaxies


@pytest.fixture(name="all_galaxies")
def make_all_galaxies(lens_galaxy, source_galaxy):
    galaxies = af.ModelInstance()
    galaxies.lens = lens_galaxy
    galaxies.source = source_galaxy
    return galaxies


@pytest.fixture(name="lens_instance")
def make_lens_instance(lens_galaxies):
    instance = af.ModelInstance()
    instance.lens_galaxies = lens_galaxies
    return instance


@pytest.fixture(name="lens_result")
def make_lens_result(lens_data_5x5, lens_instance):
    return phase_imaging.LensPlanePhase.Result(
        constant=lens_instance, figure_of_merit=1.0, previous_variable=af.ModelMapper(),
        gaussian_tuples=None,
        analysis=phase_imaging.LensPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0),
        optimizer=None)


@pytest.fixture(name="lens_source_instance")
def make_lens_source_instance(lens_galaxy, source_galaxy):
    source_galaxies = af.ModelInstance()
    lens_galaxies = af.ModelInstance()
    source_galaxies.source = source_galaxy
    lens_galaxies.lens = lens_galaxy

    instance = af.ModelInstance()
    instance.source_galaxies = source_galaxies
    instance.lens_galaxies = lens_galaxies
    return instance


@pytest.fixture(name="lens_source_result")
def make_lens_source_result(lens_data_5x5, lens_source_instance):
    return phase_imaging.LensSourcePlanePhase.Result(
        constant=lens_source_instance, figure_of_merit=1.0,
        previous_variable=af.ModelMapper(), gaussian_tuples=None,
        analysis=phase_imaging.LensSourcePlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0),
        optimizer=None)


@pytest.fixture(name="multi_plane_instance")
def make_multi_plane_instance(all_galaxies):
    instance = af.ModelInstance()
    instance.galaxies = all_galaxies
    return instance


@pytest.fixture(name="multi_plane_result")
def make_multi_plane_result(lens_data_5x5, multi_plane_instance):
    return phase_imaging.MultiPlanePhase.Result(
        constant=multi_plane_instance, figure_of_merit=1.0,
        previous_variable=af.ModelMapper(), gaussian_tuples=None,
        analysis=phase_imaging.MultiPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0),
        optimizer=None)


class TestPixelization(object):

    def test_make_pixelization_variable(self):
        instance = af.ModelInstance()
        mapper = af.ModelMapper()

        mapper.lens_galaxy = gm.GalaxyModel(redshift=g.Redshift,
                                            pixelization=px.Rectangular,
                                            regularization=rg.Constant)
        mapper.source_galaxy = gm.GalaxyModel(
            redshift=g.Redshift,
            light=lp.EllipticalLightProfile)

        assert mapper.prior_count == 9

        instance.lens_galaxy = g.Galaxy(pixelization=px.Rectangular(),
                                        regularization=rg.Constant(), redshift=1.0)
        instance.source_galaxy = g.Galaxy(redshift=1.0,
                                          light=lp.EllipticalLightProfile())

        phase_hyper.HyperPixelizationPhase.transfer_classes(instance, mapper)

        assert mapper.prior_count == 3
        assert mapper.lens_galaxy.redshift == 1.0
        assert mapper.source_galaxy.light.axis_ratio == 1.0


class TestImagePassing(object):

    def test_path_galaxy_tuples(self, lens_result, lens_galaxy):
        assert lens_result.path_galaxy_tuples == [
            (("lens_galaxies", "lens"), lens_galaxy)]

    def test_lens_source_galaxy_dict(self, lens_source_result, lens_galaxy,
                                     source_galaxy):
        assert lens_source_result.path_galaxy_tuples == [
            (("source_galaxies", "source"), source_galaxy),
            (("lens_galaxies", "lens"), lens_galaxy)
        ]

    def test_multi_plane_galaxy_dict(self, multi_plane_result, lens_galaxy,
                                     source_galaxy):
        assert multi_plane_result.path_galaxy_tuples == [
            (("galaxies", "lens"), lens_galaxy),
            (("galaxies", "source"), source_galaxy)
        ]

    def test_lens_image_dict(self, lens_result):
        image_dict = lens_result.image_2d_dict
        assert isinstance(image_dict[("lens_galaxies", "lens")], np.ndarray)

    def test_lens_source_image_dict(self, lens_source_result):
        image_dict = lens_source_result.image_2d_dict
        assert isinstance(image_dict[("lens_galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("source_galaxies", "source")], np.ndarray)

        lens_source_result.constant.lens_galaxies.lens = g.Galaxy(redshift=0.5)
        lens_source_result.constant.source_galaxies.source = g.Galaxy(redshift=1.0)

    def test_multi_plane_image_dict(self, multi_plane_result):
        image_dict = multi_plane_result.image_2d_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        multi_plane_result.constant.galaxies.lens = g.Galaxy(redshift=0.5)

        image_dict = multi_plane_result.image_2d_dict
        assert (image_dict[("galaxies", "lens")] == np.zeros((5, 5))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

    def test_galaxy_image_dict(self, lens_galaxy, source_galaxy, grid_stack_5x5,
                               convolver_image_5x5):
        tracer = rt.TracerImageSourcePlanes([lens_galaxy], [source_galaxy],
                                            grid_stack_5x5)

        assert len(tracer.galaxy_image_dict_from_convolver_image(
            convolver_image=convolver_image_5x5)) == 2
        assert lens_galaxy in tracer.galaxy_image_dict_from_convolver_image(
            convolver_image=convolver_image_5x5)
        assert source_galaxy in tracer.galaxy_image_dict_from_convolver_image(
            convolver_image=convolver_image_5x5)

    def test__results_are_passed_to_new_analysis__sets_up_hyper_images(
            self,
            mask_function_5x5,
            results_collection_5x5,
            ccd_data_5x5
    ):
        # noinspection PyPep8Naming
        Phase = phase_imaging.LensSourcePlanePhase

        phase_5x5 = Phase(lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)),
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_5x5,
            phase_name='test_phase')

        analysis = phase_5x5.make_analysis(data=ccd_data_5x5,
                                           results=results_collection_5x5)

        assert (analysis.hyper_model_image_1d == 5.0 * np.ones(9)).all()

        assert (analysis.hyper_galaxy_image_1d_path_dict[('g0',)] == 2.0 * np.ones(
            9)).all()
        assert (analysis.hyper_galaxy_image_1d_path_dict[('g1',)] == 3.0 * np.ones(
            9)).all()

    def test__results_are_passed_to_new_analysis__sets_up_hyper_cluster_images(self, mask_function_5x5, results_collection_5x5,
            ccd_data_5x5):

        # noinspection PyPep8Naming
        Phase = phase_imaging.LensSourcePlanePhase

        phase_5x5 = Phase(lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)),
                          optimizer_class=mock_pipeline.MockNLO, mask_function=mask_function_5x5,
                          cluster_pixel_scale=None, phase_name='test_phase')

        analysis = phase_5x5.make_analysis(data=ccd_data_5x5, results=results_collection_5x5)

        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g0',)] == 2.0 * np.ones(9)).all()
        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g1',)] == 3.0 * np.ones(9)).all()

        phase_5x5 = Phase(lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)),
                          optimizer_class=mock_pipeline.MockNLO, mask_function=mask_function_5x5,
                          cluster_pixel_scale=ccd_data_5x5.pixel_scale, phase_name='test_phase')

        analysis = phase_5x5.make_analysis(data=ccd_data_5x5, results=results_collection_5x5)

        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g0',)] == 2.0 * np.ones(9)).all()
        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g1',)] == 3.0 * np.ones(9)).all()
        assert len(analysis.hyper_galaxy_cluster_image_1d_path_dict[('g0',)]) == \
               analysis.lens_data.cluster.shape[0]
        assert len(analysis.hyper_galaxy_cluster_image_1d_path_dict[('g1',)]) == \
               analysis.lens_data.cluster.shape[0]

        phase_5x5 = Phase(lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)),
                          optimizer_class=mock_pipeline.MockNLO, mask_function=mask_function_5x5,
                          cluster_pixel_scale=ccd_data_5x5.pixel_scale*2.0, phase_name='test_phase')

        analysis = phase_5x5.make_analysis(data=ccd_data_5x5, results=results_collection_5x5)

        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g0',)] == 2.0 * np.ones(4)).all()
        assert (analysis.hyper_galaxy_cluster_image_1d_path_dict[('g1',)] == 3.0 * np.ones(4)).all()
        assert len(analysis.hyper_galaxy_cluster_image_1d_path_dict[('g0',)]) == \
               analysis.lens_data.cluster.shape[0]
        assert len(analysis.hyper_galaxy_cluster_image_1d_path_dict[('g1',)]) == \
               analysis.lens_data.cluster.shape[0]

    def test__image_in_results_has_masked_value_passsed__raises_error(self,
                                                                      mask_function_5x5,
                                                                      results_collection_5x5,
                                                                      ccd_data_5x5):
        # noinspection PyPep8Naming
        Phase = phase_imaging.LensSourcePlanePhase

        phase_5x5 = Phase(lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)),
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_5x5,
            phase_name='test_phase')

        results_collection_5x5[0].galaxy_images[0][2, 2] = 0.0

        with pytest.raises(exc.PhaseException):
            phase_5x5.make_analysis(data=ccd_data_5x5, results=results_collection_5x5)

    def test_associate_images_lens(self, lens_instance, lens_result, lens_data_5x5):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", lens_result)
        analysis = phase_imaging.LensPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=None, positions_threshold=None,
            results=results_collection, uses_hyper_images=True)

        instance = analysis.associate_images(instance=lens_instance)

        hyper_model_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=lens_result.image_2d_dict[("lens_galaxies", "lens")])

        assert instance.lens_galaxies.lens.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)
        assert instance.lens_galaxies.lens.hyper_galaxy_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)

    def test_associate_images_lens_source(self, lens_source_instance,
                                          lens_source_result, lens_data_5x5):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", lens_source_result)
        analysis = phase_imaging.LensSourcePlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=None, positions_threshold=None,
            results=results_collection, uses_hyper_images=True)

        instance = analysis.associate_images(lens_source_instance)

        hyper_lens_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=lens_source_result.image_2d_dict[("lens_galaxies", "lens")])
        hyper_source_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=lens_source_result.image_2d_dict[("source_galaxies", "source")])

        hyper_model_image_1d = hyper_lens_image_1d + hyper_source_image_1d

        assert instance.lens_galaxies.lens.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)
        assert instance.source_galaxies.source.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)

        assert instance.lens_galaxies.lens.hyper_galaxy_image_1d == pytest.approx(
            hyper_lens_image_1d, 1.0e-4)
        assert instance.source_galaxies.source.hyper_galaxy_image_1d == pytest.approx(
            hyper_source_image_1d, 1.04e-4)

    def test_associate_images_multi_plane(self, multi_plane_instance,
                                          multi_plane_result, lens_data_5x5):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", multi_plane_result)
        analysis = phase_imaging.MultiPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=None, positions_threshold=None,
            results=results_collection, uses_hyper_images=True)

        instance = analysis.associate_images(instance=multi_plane_instance)

        hyper_lens_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=multi_plane_result.image_2d_dict[("galaxies", "lens")])
        hyper_source_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=multi_plane_result.image_2d_dict[("galaxies", "source")])

        hyper_model_image_1d = hyper_lens_image_1d + hyper_source_image_1d

        assert instance.galaxies.lens.hyper_galaxy_image_1d == pytest.approx(
            hyper_lens_image_1d, 1.0e-4)
        assert instance.galaxies.source.hyper_galaxy_image_1d == pytest.approx(
            hyper_source_image_1d, 1.0e-4)

        assert instance.galaxies.lens.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)
        assert instance.galaxies.source.hyper_model_image_1d == pytest.approx(
            hyper_model_image_1d, 1.0e-4)

    def test_fit_uses_hyper_fit_correctly_multi_plane(self, multi_plane_instance,
                                                      multi_plane_result,
                                                      lens_data_5x5):
        results_collection = af.ResultsCollection()
        results_collection.add("phase", multi_plane_result)
        analysis = phase_imaging.MultiPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=None,
            results=results_collection, uses_hyper_images=True)

        hyper_galaxy = g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0,
                                     noise_power=1.0)

        multi_plane_instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        fit_figure_of_merit = analysis.fit(instance=multi_plane_instance)

        hyper_lens_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=multi_plane_result.image_2d_dict[("galaxies", "lens")])
        hyper_source_image_1d = lens_data_5x5.array_1d_from_array_2d(
            array_2d=multi_plane_result.image_2d_dict[("galaxies", "source")])

        hyper_model_image_1d = hyper_lens_image_1d + hyper_source_image_1d

        g0 = g.Galaxy(redshift=0.5,
                      light_profile=multi_plane_instance.galaxies.lens.light,
                      mass_profile=multi_plane_instance.galaxies.lens.mass,
                      hyper_galaxy=hyper_galaxy,
                      hyper_model_image_1d=hyper_model_image_1d,
                      hyper_galaxy_image_1d=hyper_lens_image_1d,
                      hyper_minimum_value=0.0)
        g1 = g.Galaxy(redshift=1.0,
                      light_profile=multi_plane_instance.galaxies.source.light)

        tracer = rt.TracerImageSourcePlanes(
            lens_galaxies=[g0],
            source_galaxies=[g1],
            image_plane_grid_stack=lens_data_5x5.grid_stack
        )

        fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data_5x5,
                                                       tracer=tracer,
                                                       padded_tracer=None)

        assert (fit_figure_of_merit == fit.figure_of_merit).all()

    # def test__results_are_passed_to_new_analysis__associate_images_works(
    #         self, lens_source_instance, mask_function_5x5, results_collection_5x5,
    #         ccd_data_5x5):
    #     Phase = phase_imaging.LensSourcePlanePhase
    #
    #     phase_5x5 = Phase(optimizer_class=mock_pipeline.MockNLO,
    #                       mask_function=mask_function_5x5,
    #                       phase_name='test_phase')
    #
    #     analysis = phase_5x5.make_analysis(data=ccd_data_5x5,
    #                                        results=results_collection_5x5)
    #
    #     instance = analysis.associate_images(instance=lens_source_instance)
    #
    #     assert (analysis.hyper_model_image_1d == 5.0 * np.ones(9)).all()
    #
    #     print(instance.lens_galaxies.lens.hyper_galaxy_image_1d)
    #
    #     assert (instance.lens_galaxies.lens.hyper_galaxy_image_1d == 2.0 * np.ones(
    #         9)).all()
    #     assert (instance.source_galaxies.source.hyper_galaxy_image_1d == 3.0 * np.ones(
#         9)).all()
