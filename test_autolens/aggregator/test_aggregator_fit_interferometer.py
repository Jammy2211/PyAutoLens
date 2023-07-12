import pytest
from os import path

from autoconf import conf
import autofit as af
import autolens as al

from test_autolens.aggregator.conftest import clean


@pytest.fixture(name="analysis")
def make_analysis(interferometer_7):
    def null(paths: af.DirectoryPaths, result):
        pass

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    analysis.save_results_for_aggregator = null

    return analysis


# def test__fit_interferometer_randomly_drawn_via_pdf_gen_from(analysis, samples, model):
#     path_prefix = "aggregator_fit_interferometer_gen"
#
#     database_file = path.join(conf.instance.output_path, "fit_interferometer.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     result = al.m.MockResult(model=model, samples=samples)
#
#     search = al.m.MockSearch(samples=samples, result=result)
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#     search.fit(model=model, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     fit_interferometer_agg = al.agg.FitInterferometerAgg(aggregator=agg)
#     fit_interferometer_pdf_gen = fit_interferometer_agg.randomly_drawn_via_pdf_gen_from(
#         total_samples=2
#     )
#
#     i = 0
#
#     for fit_interferometer_gen in fit_interferometer_pdf_gen:
#         for fit_interferometer in fit_interferometer_gen:
#             i += 1
#
#             assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
#             assert fit_interferometer.tracer.galaxies[0].light.centre == (10.0, 10.0)
#             assert fit_interferometer.tracer.galaxies[1].redshift == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
#
#
# def test__fit_interferometer_all_above_weight_gen(analysis, samples, model):
#     path_prefix = "aggregator_fit_interferometer_gen"
#
#     database_file = path.join(conf.instance.output_path, "fit_interferometer.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     search = al.m.MockSearch(
#         samples=samples, result=al.m.MockResult(model=model, samples=samples)
#     )
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#     search.fit(model=model, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     fit_interferometer_agg = al.agg.FitInterferometerAgg(aggregator=agg)
#     fit_interferometer_pdf_gen = fit_interferometer_agg.all_above_weight_gen_from(
#         minimum_weight=-1.0
#     )
#
#     i = 0
#
#     for fit_interferometer_gen in fit_interferometer_pdf_gen:
#         for fit_interferometer in fit_interferometer_gen:
#             i += 1
#
#             if i == 1:
#                 assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
#                 assert fit_interferometer.tracer.galaxies[0].light.centre == (1.0, 1.0)
#                 assert fit_interferometer.tracer.galaxies[1].redshift == 1.0
#
#             if i == 2:
#                 assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
#                 assert fit_interferometer.tracer.galaxies[0].light.centre == (
#                     10.0,
#                     10.0,
#                 )
#                 assert fit_interferometer.tracer.galaxies[1].redshift == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
