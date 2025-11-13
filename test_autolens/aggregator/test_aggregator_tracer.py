import autolens as al

from test_autolens.aggregator.conftest import clean, aggregator_from

database_file = "db_tracer"


def test__tracer_randomly_drawn_via_pdf_gen_from(
    masked_imaging_7x7,
    samples,
    model,
):
    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    tracer_agg = al.agg.TracerAgg(aggregator=agg)
    tracer_pdf_gen = tracer_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for tracer_gen in tracer_pdf_gen:
        for tracer_list in tracer_gen:
            i += 1

            assert tracer_list[0].galaxies[0].redshift == 0.5
            assert tracer_list[0].galaxies[0].light.centre == (10.0, 10.0)
            assert tracer_list[0].galaxies[1].redshift == 1.0

    assert i == 2

    clean(database_file=database_file)


def test__tracer_all_above_weight_gen(analysis_imaging_7x7, samples, model):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    tracer_agg = al.agg.TracerAgg(aggregator=agg)
    tracer_pdf_gen = tracer_agg.all_above_weight_gen_from(minimum_weight=-1.0)
    weight_pdf_gen = tracer_agg.weights_above_gen_from(minimum_weight=-1.0)

    i = 0

    for tracer_gen, weight_gen in zip(tracer_pdf_gen, weight_pdf_gen):
        for tracer_list in tracer_gen:
            i += 1

            if i == 1:
                assert tracer_list[0].galaxies[0].redshift == 0.5
                assert tracer_list[0].galaxies[0].light.centre == (1.0, 1.0)
                assert tracer_list[0].galaxies[1].redshift == 1.0

            if i == 2:
                assert tracer_list[0].galaxies[0].redshift == 0.5
                assert tracer_list[0].galaxies[0].light.centre == (10.0, 10.0)
                assert tracer_list[0].galaxies[1].redshift == 1.0

        for weight in weight_gen:
            if i == 0:
                assert weight == 0.0

            if i == 1:
                assert weight == 1.0

    assert i == 2

    clean(database_file=database_file)
