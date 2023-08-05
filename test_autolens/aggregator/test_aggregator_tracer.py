from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag
import autolens as al

from test_autolens.aggregator.conftest import clean


def test__tracer_randomly_drawn_via_pdf_gen_from(
    masked_imaging_7x7,
    adapt_model_image_7x7,
    adapt_galaxy_image_path_dict_7x7,
    samples,
    model,
):
    path_prefix = "aggregator_tracer_gen"

    database_file = path.join(conf.instance.output_path, "tracer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )

    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.adapt_model_image = adapt_model_image_7x7
    analysis.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict_7x7

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    tracer_agg = al.agg.TracerAgg(aggregator=agg)
    tracer_pdf_gen = tracer_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for tracer_gen in tracer_pdf_gen:
        for tracer in tracer_gen:
            i += 1

            assert tracer.galaxies[0].redshift == 0.5
            assert tracer.galaxies[0].light.centre == (10.0, 10.0)
            assert tracer.galaxies[1].redshift == 1.0

    assert i == 2

    clean(database_file=database_file, result_path=result_path)


# def test__tracer_all_above_weight_gen(masked_imaging_7x7, samples, model):
#     path_prefix = "aggregator_tracer_gen"
#
#     database_file = path.join(conf.instance.output_path, "tracer.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     search = al.m.MockSearch(
#         samples=samples, result=al.m.MockResult(model=model, samples=samples)
#     )
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#     analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
#     search.fit(model=model, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     tracer_agg = al.agg.TracerAgg(aggregator=agg)
#     tracer_pdf_gen = tracer_agg.all_above_weight_gen_from(minimum_weight=-1.0)
#     weight_pdf_gen = tracer_agg.weights_above_gen_from(minimum_weight=-1.0)
#
#     i = 0
#
#     for tracer_gen, weight_gen in zip(tracer_pdf_gen, weight_pdf_gen):
#         for tracer in tracer_gen:
#             i += 1
#
#             if i == 1:
#                 assert tracer.galaxies[0].redshift == 0.5
#                 assert tracer.galaxies[0].light.centre == (1.0, 1.0)
#                 assert tracer.galaxies[1].redshift == 1.0
#
#             if i == 2:
#                 assert tracer.galaxies[0].redshift == 0.5
#                 assert tracer.galaxies[0].light.centre == (10.0, 10.0)
#                 assert tracer.galaxies[1].redshift == 1.0
#
#         for weight in weight_gen:
#             if i == 0:
#                 assert weight == 0.0
#
#             if i == 1:
#                 assert weight == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
