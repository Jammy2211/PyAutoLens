from os import path

from autoconf import conf
import autofit as af
import autolens as al

from test_autolens.aggregator.conftest import clean


# def test__fit_imaging_randomly_drawn_via_pdf_gen_from(analysis, samples, model):
#     path_prefix = "aggregator_fit_imaging_gen"
#
#     database_file = path.join(conf.instance.output_path, "fit_imaging.sqlite")
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
#     fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg)
#     fit_imaging_pdf_gen = fit_imaging_agg.randomly_drawn_via_pdf_gen_from(
#         total_samples=2
#     )
#
#     i = 0
#
#     for fit_imaging_gen in fit_imaging_pdf_gen:
#         for fit_imaging in fit_imaging_gen:
#             i += 1
#
#             assert fit_imaging.tracer.galaxies[0].redshift == 0.5
#             assert fit_imaging.tracer.galaxies[0].light.centre == (10.0, 10.0)
#             assert fit_imaging.tracer.galaxies[1].redshift == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
#
#
# def test__fit_imaging_all_above_weight_gen(analysis, samples, model):
#     path_prefix = "aggregator_fit_imaging_gen"
#
#     database_file = path.join(conf.instance.output_path, "fit_imaging.sqlite")
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
#     fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg)
#     fit_imaging_pdf_gen = fit_imaging_agg.all_above_weight_gen_from(minimum_weight=-1.0)
#
#     i = 0
#
#     for fit_imaging_gen in fit_imaging_pdf_gen:
#         for fit_imaging in fit_imaging_gen:
#             i += 1
#
#             if i == 1:
#                 assert fit_imaging.tracer.galaxies[0].redshift == 0.5
#                 assert fit_imaging.tracer.galaxies[0].light.centre == (1.0, 1.0)
#                 assert fit_imaging.tracer.galaxies[1].redshift == 1.0
#
#             if i == 2:
#                 assert fit_imaging.tracer.galaxies[0].redshift == 0.5
#                 assert fit_imaging.tracer.galaxies[0].light.centre == (10.0, 10.0)
#                 assert fit_imaging.tracer.galaxies[1].redshift == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
