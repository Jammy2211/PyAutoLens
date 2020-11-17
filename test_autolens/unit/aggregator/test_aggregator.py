from os import path

import autofit as af
import autolens as al
import pytest
from autolens.mock import mock

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


@pytest.fixture(name="samples")
def make_samples():

    galaxy_0 = al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(centre=(0.0, 1.0)))
    galaxy_1 = al.Galaxy(redshift=1.0, light=al.lp.EllipticalSersic())

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

    return mock.MockSamples(max_log_likelihood_instance=tracer)


def test__tracer_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = al.PhaseImaging(
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=mock.MockSearch("test_phase_aggregator", samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    tracer_gen = al.agg.Tracer(aggregator=agg)

    for tracer in tracer_gen:

        assert tracer.galaxies[0].redshift == 0.5
        assert tracer.galaxies[0].light.centre == (0.0, 1.0)
        assert tracer.galaxies[1].redshift == 1.0


def test__masked_imaging_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = al.PhaseImaging(
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        settings=al.SettingsPhaseImaging(
            settings_masked_imaging=al.SettingsMaskedImaging(
                grid_class=al.GridIterate,
                grid_inversion_class=al.GridInterpolate,
                fractional_accuracy=0.5,
                sub_steps=[2],
                pixel_scales_interp=0.1,
            )
        ),
        search=mock.MockSearch("test_phase_aggregator", samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()
        assert isinstance(masked_imaging.grid, al.GridIterate)
        assert isinstance(masked_imaging.grid_inversion, al.GridInterpolate)
        assert masked_imaging.grid.sub_steps == [2]
        assert masked_imaging.grid.fractional_accuracy == 0.5
        assert masked_imaging.grid_inversion.pixel_scales_interp == (0.1, 0.1)


def test__fit_imaging_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = al.PhaseImaging(
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=mock.MockSearch("test_phase_aggregator", samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    fit_imaging_gen = al.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.masked_imaging.imaging.image == imaging_7x7.image).all()


def test__masked_interferometer_generator_from_aggregator(
    interferometer_7, mask_7x7, samples
):

    phase_interferometer_7x7 = al.PhaseInterferometer(
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        settings=al.SettingsPhaseInterferometer(
            settings_masked_interferometer=al.SettingsMaskedInterferometer(
                transformer_class=al.TransformerDFT,
                grid_class=al.GridIterate,
                grid_inversion_class=al.GridInterpolate,
                fractional_accuracy=0.5,
                sub_steps=[2],
                pixel_scales_interp=0.1,
            )
        ),
        search=mock.MockSearch("test_phase_aggregator", samples=samples),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=mask_7x7,
        results=mock.MockResults(samples=samples),
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    masked_interferometer_gen = al.agg.MaskedInterferometer(aggregator=agg)

    for masked_interferometer in masked_interferometer_gen:
        assert (
            masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer.real_space_mask == mask_7x7).all()
        assert isinstance(masked_interferometer.grid, al.GridIterate)
        assert isinstance(masked_interferometer.grid_inversion, al.GridInterpolate)
        assert masked_interferometer.grid.sub_steps == [2]
        assert masked_interferometer.grid.fractional_accuracy == 0.5
        assert masked_interferometer.grid_inversion.pixel_scales_interp == (0.1, 0.1)
        assert isinstance(masked_interferometer.transformer, al.TransformerDFT)


def test__fit_interferometer_generator_from_aggregator(
    interferometer_7, mask_7x7, samples
):

    phase_interferometer_7x7 = al.PhaseInterferometer(
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=mock.MockSearch("test_phase_aggregator", samples=samples),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=mask_7x7,
        results=mock.MockResults(samples=samples),
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    fit_interferometer_gen = al.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (
            fit_interferometer.masked_interferometer.real_space_mask == mask_7x7
        ).all()


class MockResult:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood
        self.log_evidence_values = log_likelihood
        self.model = log_likelihood


class MockAggregator:
    def __init__(self, grid_search_result):

        self.grid_search_result = grid_search_result

    @property
    def grid_search_results(self):
        return iter([self.grid_search_result])

    def values(self, str):
        return self.grid_search_results


# def test__results_array_from_results_file(path):
#
#     results = [
#         MockResult(log_likelihood=1.0),
#         MockResult(log_likelihood=2.0),
#         MockResult(log_likelihood=3.0),
#         MockResult(log_likelihood=4.0),
#     ]
#
#     lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
#     physical_lower_limits_lists = [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
#
#     grid_search_result = af.GridSearchResult(
#         results=results,
#         physical_lower_limits_lists=physical_lower_limits_lists,
#         lower_limit_lists=lower_limit_lists,
#     )
#
#     aggregator = MockAggregator(grid_search_result=grid_search_result)
#
#     array = al.agg.grid_search_result_as_array(aggregator=aggregator)
#
#     assert array.in_2d == pytest.approx(np.array([[3.0, 2.0], [1.0, 4.0]]), 1.0e4)
#     assert array.pixel_scales == (1.0, 1.0)


# def test__results_array_from_real_grid_search_pickle(path):
#
#     with open("{}/{}.pickle".format(path, "grid_search_result"), "rb") as f:
#         grid_search_result = pickle.load(f)
#
#     array = al.agg.grid_search_log_evidences_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.in_2d)
#
#     array = al.agg.grid_search_subhalo_masses_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.in_2d)
#
#     array = al.agg.grid_search_subhalo_centres_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array)
