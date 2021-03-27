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


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.EllipticalSersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.EllipticalSersic),
        )
    )


def test__tracer_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(
        paths=af.Paths(path_prefix="aggregator_tracer_gen"), samples=samples
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    tracer_gen = al.agg.Tracer(aggregator=agg)

    for tracer in tracer_gen:

        assert tracer.galaxies[0].redshift == 0.5
        assert tracer.galaxies[0].light.centre == (0.0, 1.0)
        assert tracer.galaxies[1].redshift == 1.0


def test__masked_imaging_generator_from_aggregator(
    imaging_7x7, mask_7x7, samples, model
):

    masked_imaging_7x7 = al.MaskedImaging(
        imaging=imaging_7x7,
        mask=mask_7x7,
        settings=al.SettingsMaskedImaging(
            grid_class=al.Grid2DIterate,
            grid_inversion_class=al.Grid2DInterpolate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            pixel_scales_interp=0.1,
        ),
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    search = mock.MockSearch(
        paths=af.Paths(path_prefix="aggregator_masked_imaging_gen"), samples=samples
    )

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()
        assert isinstance(masked_imaging.grid, al.Grid2DIterate)
        assert isinstance(masked_imaging.grid_inversion, al.Grid2DInterpolate)
        assert masked_imaging.grid.sub_steps == [2]
        assert masked_imaging.grid.fractional_accuracy == 0.5
        assert masked_imaging.grid_inversion.pixel_scales_interp == (0.1, 0.1)


def test__fit_imaging_generator_from_aggregator(masked_imaging_7x7, samples, model):

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    search = mock.MockSearch(
        paths=af.Paths(path_prefix="aggregator_fit_imaging_gen"), samples=samples
    )

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_imaging_gen = al.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (
            fit_imaging.masked_imaging.imaging.image == masked_imaging_7x7.imaging.image
        ).all()


def test__masked_interferometer_generator_from_aggregator(
    interferometer_7, visibilities_mask_7, mask_7x7, samples, model
):

    masked_interferometer = al.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7,
        real_space_mask=mask_7x7,
        settings=al.SettingsMaskedInterferometer(
            transformer_class=al.TransformerDFT,
            grid_class=al.Grid2DIterate,
            grid_inversion_class=al.Grid2DInterpolate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            pixel_scales_interp=0.1,
        ),
    )

    search = mock.MockSearch(
        paths=af.Paths(path_prefix="aggregator_masked_interferometer_gen"),
        samples=samples,
    )

    analysis = al.AnalysisInterferometer(dataset=masked_interferometer)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    masked_interferometer_gen = al.agg.MaskedInterferometer(aggregator=agg)

    for masked_interferometer in masked_interferometer_gen:
        assert (
            masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer.real_space_mask == mask_7x7).all()
        assert isinstance(masked_interferometer.grid, al.Grid2DIterate)
        assert isinstance(masked_interferometer.grid_inversion, al.Grid2DInterpolate)
        assert masked_interferometer.grid.sub_steps == [2]
        assert masked_interferometer.grid.fractional_accuracy == 0.5
        assert masked_interferometer.grid_inversion.pixel_scales_interp == (0.1, 0.1)
        assert isinstance(masked_interferometer.transformer, al.TransformerDFT)


def test__fit_interferometer_generator_from_aggregator(
    masked_interferometer_7, mask_7x7, samples, model
):

    search = mock.MockSearch(
        paths=af.Paths(path_prefix="aggregator_fit_interferometer_gen"), samples=samples
    )

    analysis = al.AnalysisInterferometer(dataset=masked_interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_interferometer_gen = al.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.masked_interferometer.interferometer.visibilities
            == masked_interferometer_7.interferometer.visibilities
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
#     assert array.native == pytest.approx(np.array([[3.0, 2.0], [1.0, 4.0]]), 1.0e4)
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
#     print(array.native)
#
#     array = al.agg.grid_search_subhalo_masses_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.native)
#
#     array = al.agg.grid_search_subhalo_centres_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array)
