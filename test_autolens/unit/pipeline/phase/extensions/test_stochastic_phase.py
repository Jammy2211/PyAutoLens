import autofit as af
import autofit.non_linear.paths
import autolens as al
import pytest
from test_autolens import mock


class MockPhase:
    def __init__(self):
        self.phase_name = "phase_name"
        self.paths = autofit.non_linear.paths.Paths(
            name=self.phase_name, path_prefix="phase_path", folders=("",), tag=""
        )
        self.search = mock.MockSearch(paths=self.paths)
        self.model = af.ModelMapper()
        self.settings = al.PhaseSettingsImaging(log_likelihood_cap=None)

    def save_dataset(self, dataset):
        pass

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def run(self, *args, **kwargs):
        result = mock.MockResult()
        result.settings = self.settings
        return result

@pytest.fixture(name="stochastic")
def make_stochastic():
    normal_phase = MockPhase()

    # noinspection PyUnusedLocal
    def run_hyper(*args, **kwargs):
        return mock.MockResult()

    return al.StochasticPhase(
        phase=normal_phase,
        search=mock.MockSearch(),
        model_classes=(al.mp.MassProfile,)
    )


class TestStochasticPhase:
    def test_stochastic_result(self, imaging_7x7, stochastic):
        result = stochastic.run(dataset=None, mask=None)

        assert hasattr(result, "stochastic")
        assert isinstance(result.stochastic, mock.MockResult)
        assert stochastic.hyper_name == "stochastic"
        assert isinstance(stochastic, al.StochasticPhase)

        # noinspection PyUnusedLocal
        def run_hyper(*args, **kwargs):
            return mock.MockResult()

        stochastic.run_hyper = run_hyper

        result = stochastic.run(dataset=imaging_7x7)

        assert hasattr(result, "stochastic")
        assert isinstance(result.stochastic, mock.MockResult)

    def test__stochastic_phase_analysis_inherits_log_likelihood_cap(self, imaging_7x7, stochastic):

        result = stochastic.run_hyper(
            dataset=imaging_7x7,
            results=mock.MockResults(stochastic_log_evidences=[1.0, 1.0, 2.0])
        )

        assert result.settings.log_likelihood_cap == 1.0


    def test__paths(self):

        galaxy = ag.Galaxy(
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
            redshift=1.0,
        )

        phase = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(galaxy=galaxy),
            search=af.DynestyStatic(n_live_points=1),
            settings=ag.PhaseSettingsImaging(bin_up_factor=2),
        )

        phase_extended = phase.extend_with_inversion_phase(
            inversion_search=af.DynestyStatic(n_live_points=1)
        )

        hyper_phase = phase_extended.make_hyper_phase()

        assert (
            "test_phase/inversion__settings__grid_sub_2__bin_2/dynesty_static__nlive_1"
            in hyper_phase.paths.output_path
        )

        phase_extended = phase.extend_with_multiple_hyper_phases(
            setup=ag.PipelineSetup(
                hyper_galaxies=True,
                inversion_search=af.DynestyStatic(n_live_points=1),
                hyper_galaxies_search=af.DynestyStatic(n_live_points=2),
                hyper_combined_search=af.DynestyStatic(n_live_points=3),
            ),
            include_inversion=True,
        )

        inversion_phase = phase_extended.hyper_phases[0].make_hyper_phase()

        assert (
            "test_phase/inversion__settings__grid_sub_2__bin_2/dynesty_static__nlive_1"
            in inversion_phase.paths.output_path
        )

        hyper_galaxy_phase = phase_extended.hyper_phases[1].make_hyper_phase()

        assert (
            "test_phase/hyper_galaxy__settings__grid_sub_2__bin_2/dynesty_static__nlive_2"
            in hyper_galaxy_phase.paths.output_path
        )

        hyper_combined_phase = phase_extended.make_hyper_phase()

        assert (
            "test_phase/hyper_combined__settings__grid_sub_2__bin_2/dynesty_static__nlive_3"
            in hyper_combined_phase.paths.output_path
        )


class TestHyperGalaxyPhase:
    def test__likelihood_function_is_same_as_normal_phase_likelihood_function(
        self, imaging_7x7, mask_7x7
    ):

        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(galaxy=galaxy),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings=ag.PhaseSettingsImaging(sub_size=2),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])

        mask = phase_imaging_7x7.meta_dataset.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        assert mask.sub_size == 2

        masked_imaging = ag.MaskedImaging(imaging=imaging_7x7, mask=mask)
        plane = analysis.plane_for_instance(instance=instance)
        fit = FitImaging(
            masked_imaging=masked_imaging,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        phase_imaging_7x7_hyper = phase_imaging_7x7.extend_with_multiple_hyper_phases(
            setup=ag.PipelineSetup(
                hyper_galaxies=True, hyper_galaxies_search=mock.MockSearch()
            )
        )

        instance = phase_imaging_7x7_hyper.model.instance_from_unit_vector([])

        instance.hyper_galaxy = ag.HyperGalaxy(noise_factor=0.0)

        analysis = phase_imaging_7x7_hyper.hyper_phases[0].Analysis(
            masked_imaging=masked_imaging,
            hyper_model_image=fit.model_image,
            hyper_galaxy_image=fit.model_image,
            image_path="files/",
        )

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=ag.HyperGalaxy(noise_factor=0.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.log_likelihood == fit.log_likelihood

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=ag.HyperGalaxy(noise_factor=1.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.log_likelihood != fit.log_likelihood

        instance.hyper_galaxy = ag.HyperGalaxy(noise_factor=0.0)

        log_likelihood = analysis.log_likelihood_function(instance=instance)

        assert log_likelihood == fit.log_likelihood
