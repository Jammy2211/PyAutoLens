import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
import autolens as al
from autolens import exc
from test.unit.mock.pipeline import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/config/phase_imaging_7x7".format(directory)
    )


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.data_path = directory


class TestPhase(object):
    def test__set_constants(self, phase_data_7x7):
        phase_data_7x7.galaxies = [al.Galaxy(redshift=0.5)]
        assert phase_data_7x7.optimizer.variable.galaxies == [al.Galaxy(redshift=0.5)]

    def test__set_variables(self, phase_data_7x7):
        phase_data_7x7.galaxies = [al.GalaxyModel(redshift=0.5)]
        assert phase_data_7x7.optimizer.variable.galaxies == [
            al.GalaxyModel(redshift=0.5)
        ]

    def test__customize(
        self, mask_function_7x7, results_7x7, results_collection_7x7, imaging_data_7x7
    ):
        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.constant.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_imaging_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )
        phase_imaging_7x7.customize_priors(results_collection_7x7)

        assert phase_imaging_7x7.galaxies == [galaxy]

        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.variable.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_imaging_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )
        phase_imaging_7x7.customize_priors(results_collection_7x7)

        assert phase_imaging_7x7.galaxies == [galaxy_model]

    def test__duplication(self):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        al.PhaseImaging(phase_name="test_phase")

        assert phase_imaging_7x7.galaxies is not None

    #
    # def test__uses_pixelization_preload_grids_if_possible(
    #     self, imaging_data_7x7, mask_function_7x7
    # ):
    #     phase_imaging_7x7 = al.PhaseImaging(
    #         phase_name="test_phase", mask_function=mask_function_7x7
    #     )
    #
    #     analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
    #
    #     galaxy = al.Galaxy(redshift=0.5)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy, galaxy], grid=analysis.lens_imaging_data.grid
    #     )
    #
    #     assert (preload_pixelization_grid.pixelization == np.array([[0.0, 0.0]])).all()
    #
    #     galaxy_pix_which_doesnt_use_pix_grid = al.Galaxy(
    #         redshift=0.5, pixelization=al.pixelizations.Rectangular(), regularization=al.regularization.Constant()
    #     )
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_doesnt_use_pix_grid],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (preload_pixelization_grid.pixelization == np.array([[0.0, 0.0]])).all()
    #
    #     galaxy_pix_which_uses_pix_grid = al.Galaxy(
    #         redshift=0.5,
    #         pixelization=al.pixelizations.VoronoiMagnification(),
    #         regularization=al.regularization.Constant(),
    #     )
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_uses_pix_grid],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (
    #         preload_pixelization_grid.pixelization
    #         == np.array(
    #             [
    #                 [1.0, -1.0],
    #                 [1.0, 0.0],
    #                 [1.0, 1.0],
    #                 [0.0, -1.0],
    #                 [0.0, 0.0],
    #                 [0.0, 1.0],
    #                 [-1.0, -1.0],
    #                 [-1.0, 0.0],
    #                 [-1.0, 1.0],
    #             ]
    #         )
    #     ).all()
    #
    #     galaxy_pix_which_uses_brightness = al.Galaxy(
    #         redshift=0.5,
    #         pixelization=al.pixelizations.VoronoiBrightnessImage(pixels=9),
    #         regularization=al.regularization.Constant(),
    #     )
    #
    #     galaxy_pix_which_uses_brightness.hyper_galaxy_cluster_image_1d = np.array(
    #         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    #     )
    #
    #     phase_imaging_7x7 = al.PhaseImaging(
    #         phase_name="test_phase",
    #         galaxies=dict(
    #             lens=al.GalaxyModel(
    #                 redshift=0.5,
    #                 pixelization=al.pixelizations.VoronoiBrightnessImage,
    #                 regularization=al.regularization.Constant,
    #             )
    #         ),
    #         inversion_pixel_limit=5,
    #         mask_function=mask_function_7x7,
    #     )
    #
    #     analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_uses_brightness],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (
    #         preload_pixelization_grid.pixelization
    #         == np.array(
    #             [
    #                 [0.0, 1.0],
    #                 [1.0, -1.0],
    #                 [-1.0, -1.0],
    #                 [-1.0, 1.0],
    #                 [0.0, -1.0],
    #                 [1.0, 1.0],
    #                 [-1.0, 0.0],
    #                 [0.0, 0.0],
    #                 [1.0, 0.0],
    #             ]
    #         )
    #     ).all()

    def test__tracer_for_instance__includes_cosmology(
        self, imaging_data_7x7, mask_function_7x7
    ):
        lens_galaxy = al.Galaxy(redshift=0.5)
        source_galaxy = al.Galaxy(redshift=0.5)

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.cosmology == cosmo.FLRW

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy, source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.source_plane.galaxies[0] == source_galaxy
        assert tracer.cosmology == cosmo.FLRW

        galaxy_0 = al.Galaxy(redshift=0.1)
        galaxy_1 = al.Galaxy(redshift=0.2)
        galaxy_2 = al.Galaxy(redshift=0.3)

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[galaxy_0, galaxy_1, galaxy_2],
            cosmology=cosmo.WMAP7,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.planes[0].galaxies[0] == galaxy_0
        assert tracer.planes[1].galaxies[0] == galaxy_1
        assert tracer.planes[2].galaxies[0] == galaxy_2
        assert tracer.cosmology == cosmo.WMAP7

    def test__phase_can_receive_list_of_galaxy_models(self):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.light_profiles.EllipticalSersic,
                    sis=al.mass_profiles.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mass_profiles.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        for item in phase_imaging_7x7.variable.path_priors_tuples:
            print(item)

        sersic = phase_imaging_7x7.variable.galaxies[0].sersic
        sis = phase_imaging_7x7.variable.galaxies[0].sis
        lens_1_sis = phase_imaging_7x7.variable.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.2,
            sersic.centre[1]: 0.2,
            sersic.axis_ratio: 0.0,
            sersic.phi: 0.1,
            sersic.effective_radius.priors[0]: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity.priors[0]: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            sis.einstein_radius.priors[0]: 0.3,
            phase_imaging_7x7.variable.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_imaging_7x7.variable.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_imaging_7x7.optimizer.variable.instance_for_arguments(
            arguments=arguments
        )

        assert instance.galaxies[0].sersic.centre[0] == 0.2
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 0.3
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

        class LensPlanePhase2(al.PhaseImaging):
            # noinspection PyUnusedLocal
            def pass_models(self, results):
                self.galaxies[0].sis.einstein_radius = 10.0

        phase_imaging_7x7 = LensPlanePhase2(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.light_profiles.EllipticalSersic,
                    sis=al.mass_profiles.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mass_profiles.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        # noinspection PyTypeChecker
        phase_imaging_7x7.pass_models(None)

        sersic = phase_imaging_7x7.variable.galaxies[0].sersic
        sis = phase_imaging_7x7.variable.galaxies[0].sis
        lens_1_sis = phase_imaging_7x7.variable.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.01,
            sersic.centre[1]: 0.2,
            sersic.axis_ratio: 0.0,
            sersic.phi: 0.1,
            sersic.effective_radius.priors[0]: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity.priors[0]: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            phase_imaging_7x7.variable.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_imaging_7x7.variable.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_imaging_7x7.optimizer.variable.instance_for_arguments(
            arguments
        )

        assert instance.galaxies[0].sersic.centre[0] == 0.01
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 10.0
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8


class TestResult(object):
    def test__results_of_phase_are_available_as_properties(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                )
            ],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert isinstance(result, al.AbstractPhase.Result)

    def test__most_likely_tracer_available_as_result(
        self, imaging_data_7x7, mask_function_7x7
    ):

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    light=al.light_profiles.EllipticalCoreSersic(intensity=2.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert isinstance(result.most_likely_tracer, al.Tracer)
        assert result.most_likely_tracer.galaxies[0].light.intensity == 1.0
        assert result.most_likely_tracer.galaxies[1].light.intensity == 2.0


class TestPhasePickle(object):

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_data_7x7, mask_function_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.MockAnalysis(1, 1)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            data=imaging_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            data=imaging_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.light_profiles.EllipticalLightProfile()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )
        phase_imaging_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_imaging_7x7.run(data_type=imaging_data_7x7, results=None, mask=None, positions=None)
