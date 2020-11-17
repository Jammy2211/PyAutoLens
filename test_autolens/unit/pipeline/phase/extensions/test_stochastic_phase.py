from os import path
import autofit as af
import autofit.non_linear.paths
import autolens as al
import pytest
from autolens.mock import mock


class MockMetaDataset:
    def __init__(self, settings):

        self.settings = settings


class MockPhase:
    def __init__(self):
        self.name = "name"
        self.paths = autofit.non_linear.paths.Paths(
            name=self.name, path_prefix="phase_path", tag=""
        )
        self.search = mock.MockSearch(paths=self.paths)
        self.model = af.ModelMapper()
        self.settings = al.SettingsPhaseImaging(log_likelihood_cap=None)
        self.meta_dataset = MockMetaDataset(settings=self.settings)

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
        hyper_search=mock.MockSearch(),
        model_classes=(al.mp.MassProfile,),
    )


class _TestStochasticPhase:
    def test__stochastic_result(self, imaging_7x7, stochastic):

        results = mock.MockResults(stochastic_log_evidences=[1.0, 1.0, 2.0])

        result = stochastic.run(dataset=None, mask=None, results=results)

        assert hasattr(result, "stochastic")
        assert isinstance(result.stochastic, mock.MockResult)
        assert stochastic.hyper_name == "stochastic"
        assert isinstance(stochastic, al.StochasticPhase)

        # noinspection PyUnusedLocal
        def run_hyper(*args, **kwargs):
            return mock.MockResult()

        stochastic.run_hyper = run_hyper

        result = stochastic.run(dataset=imaging_7x7, results=results)

        assert hasattr(result, "stochastic")
        assert isinstance(result.stochastic, mock.MockResult)

    def test__stochastic_phase_analysis_inherits_log_likelihood_cap(
        self, imaging_7x7, stochastic
    ):

        result = stochastic.run_hyper(
            dataset=imaging_7x7,
            results=mock.MockResults(stochastic_log_evidences=[1.0, 1.0, 2.0]),
        )

        assert result.settings.log_likelihood_cap == 1.0

    def test__paths(self):

        galaxy = al.Galaxy(mass=al.mp.SphericalIsothermal(), redshift=1.0)

        phase = al.PhaseImaging(
            galaxies=dict(galaxy=galaxy),
            search=af.DynestyStatic(name="test_phase", n_live_points=1),
            settings=al.SettingsPhaseImaging(bin_up_factor=2),
        )

        phase_stochastic = phase.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=1)
        )

        hyper_phase = phase_stochastic.make_hyper_phase()

        assert (
            path.join(
                "test_phase",
                "stochastic__settings__grid_sub_2__bin_2",
                "dynesty_static__nlive_1",
            )
            in hyper_phase.paths.output_path
        )
