import autolens as al

from test_autolens.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_imaging"


def test__fit_imaging_randomly_drawn_via_pdf_gen_from__analysis_single(
    analysis_imaging_7x7, samples, model
):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = al.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].tracer.galaxies[0].redshift == 0.5
            assert fit_list[0].tracer.galaxies[0].light.centre == (10.0, 10.0)
            assert fit_list[0].tracer.galaxies[1].redshift == 1.0

            assert fit_list[0].dataset_model.background_sky_level == 10.0

    assert i == 2

    clean(database_file=database_file)


# TODO : These need to use FactorGraphModel

# def test__fit_imaging_randomly_drawn_via_pdf_gen_from__analysis_multi(
#     analysis_imaging_7x7, samples, model
# ):
#     agg = aggregator_from(
#         database_file=database_file,
#         analysis=analysis_imaging_7x7 + analysis_imaging_7x7,
#         model=model,
#         samples=samples,
#     )
#
#     fit_agg = al.agg.FitImagingAgg(aggregator=agg)
#     fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)
#
#     i = 0
#
#     for fit_gen in fit_pdf_gen:
#         for fit_list in fit_gen:
#             i += 1
#
#             assert fit_list[0].tracer.galaxies[0].redshift == 0.5
#             assert fit_list[0].tracer.galaxies[0].light.centre == (10.0, 10.0)
#             assert fit_list[0].tracer.galaxies[1].redshift == 1.0
#
#             assert fit_list[1].tracer.galaxies[0].redshift == 0.5
#             assert fit_list[1].tracer.galaxies[0].light.centre == (10.0, 10.0)
#             assert fit_list[1].tracer.galaxies[1].redshift == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file)


def test__fit_imaging_all_above_weight_gen(analysis_imaging_7x7, samples, model):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = al.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            if i == 1:
                assert fit_list[0].tracer.galaxies[0].redshift == 0.5
                assert fit_list[0].tracer.galaxies[0].light.centre == (1.0, 1.0)
                assert fit_list[0].tracer.galaxies[1].redshift == 1.0

            if i == 2:
                assert fit_list[0].tracer.galaxies[0].redshift == 0.5
                assert fit_list[0].tracer.galaxies[0].light.centre == (10.0, 10.0)
                assert fit_list[0].tracer.galaxies[1].redshift == 1.0

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging__adapt_images(
    analysis_imaging_7x7, samples, model, adapt_images_7x7
):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = al.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1
            assert (
                list(fit_list[0].adapt_images.galaxy_image_dict.values())[0]
                == list(adapt_images_7x7.galaxy_name_image_dict.values())[0]
            ).all()

            assert (
                list(
                    fit_list[0].adapt_images.galaxy_image_plane_mesh_grid_dict.values()
                )[0]
                == list(
                    adapt_images_7x7.galaxy_name_image_plane_mesh_grid_dict.values()
                )[0]
            ).all()

    assert i == 2

    clean(database_file=database_file)
