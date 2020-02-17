import os
from os import path

import pytest

import autofit as af
import autolens as al
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/../test_files/config/".format(directory))

    af.conf.instance = af.conf.Config("{}/../../test_files/config/".format(directory))


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.dataset_path = directory


class TestPhase:

    # TODO : These tests have both turned to models?

    def test__set_instances(self, phase_dataset_7x7):
        phase_dataset_7x7.galaxies = [al.Galaxy(redshift=0.5)]
        assert phase_dataset_7x7.model.galaxies == [al.Galaxy(redshift=0.5)]

    def test__set_models(self, phase_dataset_7x7):
        phase_dataset_7x7.galaxies = [al.GalaxyModel(redshift=0.5)]
        assert phase_dataset_7x7.model.galaxies == [al.GalaxyModel(redshift=0.5)]

    def test__promise_attrbutes(self):
        phase = al.PhaseDataset(
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    mass=al.mp.EllipticalIsothermal,
                    shear=al.mp.ExternalShear,
                ),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            optimizer_class=mock_pipeline.MockNLO,
            phase_tag="",
            phase_name="test_phase",
        )

        print(hasattr(af.last.result.instance.galaxies.lens, "mas2s"))

    def test__customize(
        self, results_7x7, results_collection_7x7, imaging_7x7, mask_7x7
    ):
        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.instance.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.instance, "galaxies", [galaxy])
        setattr(results_7x7.model, "galaxies", [galaxy_model])

        phase_dataset_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase", optimizer_class=mock_pipeline.MockNLO
        )

        phase_dataset_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results_collection_7x7
        )
        phase_dataset_7x7.customize_priors(results_collection_7x7)

        assert phase_dataset_7x7.galaxies == [galaxy]

        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.model.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.instance, "galaxies", [galaxy])
        setattr(results_7x7.model, "galaxies", [galaxy_model])

        phase_dataset_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase", optimizer_class=mock_pipeline.MockNLO
        )

        phase_dataset_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results_collection_7x7
        )
        phase_dataset_7x7.customize_priors(results_collection_7x7)

        assert phase_dataset_7x7.galaxies == [galaxy_model]

    def test__duplication(self):
        phase_dataset_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        al.PhaseImaging(phase_name="test_phase")

        assert phase_dataset_7x7.galaxies is not None

    def test__uses_pixelization_preload_grids_if_possible(self, imaging_7x7, mask_7x7):
        phase_dataset_7x7 = al.PhaseImaging(phase_name="test_phase")

        analysis = phase_dataset_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)

        assert analysis.masked_dataset.preload_sparse_grids_of_planes is None

    def test__phase_can_receive_list_of_galaxy_models(self):
        phase_dataset_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.lp.EllipticalSersic,
                    sis=al.mp.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mp.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        for item in phase_dataset_7x7.model.path_priors_tuples:
            print(item)

        sersic = phase_dataset_7x7.model.galaxies[0].sersic
        sis = phase_dataset_7x7.model.galaxies[0].sis
        lens_1_sis = phase_dataset_7x7.model.galaxies[1].sis

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
            phase_dataset_7x7.model.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_dataset_7x7.model.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments=arguments)

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

        phase_dataset_7x7 = LensPlanePhase2(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.lp.EllipticalSersic,
                    sis=al.mp.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mp.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        # noinspection PyTypeChecker
        phase_dataset_7x7.pass_models(None)

        sersic = phase_dataset_7x7.model.galaxies[0].sersic
        sis = phase_dataset_7x7.model.galaxies[0].sis
        lens_1_sis = phase_dataset_7x7.model.galaxies[1].sis

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
            phase_dataset_7x7.model.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_dataset_7x7.model.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments)

        assert instance.galaxies[0].sersic.centre[0] == 0.01
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 10.0
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):
        clean_images()

        phase_dataset_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=[
                al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))
            ],
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(dataset=imaging_7x7, mask=mask_7x7)

        assert isinstance(result, al.AbstractPhase.Result)

    def test__most_likely_tracer_available_as_result(self, imaging_7x7, mask_7x7):

        phase_dataset_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0, light=al.lp.EllipticalCoreSersic(intensity=2.0)
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(dataset=imaging_7x7, mask=mask_7x7)

        assert isinstance(result.most_likely_tracer, al.Tracer)
        assert result.most_likely_tracer.galaxies[0].light.intensity == 1.0
        assert result.most_likely_tracer.galaxies[1].light.intensity == 2.0


class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.GalaxiesMockAnalysis(1, 1)

        phase_dataset_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_dataset_7x7.make_analysis = make_analysis
        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        phase_dataset_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_dataset_7x7.make_analysis = make_analysis
        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.lp.EllipticalLightProfile()

        phase_dataset_7x7 = CustomPhase(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )
        phase_dataset_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_dataset_7x7.run(data_type=imaging_7x7, results=None, mask=None, positions=None)
