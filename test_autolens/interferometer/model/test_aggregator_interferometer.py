from os import path
import os
import pytest
import shutil

from autoconf import conf
import autofit as af
import autolens as al
from autofit.non_linear.samples import Sample

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.EllSersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.EllSersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(centre=(0.0, 1.0)))
    galaxy_1 = al.Galaxy(redshift=1.0, light=al.lp.EllSersic())

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return al.m.MockSamples(
        model=model, sample_list=sample_list, max_log_likelihood_instance=tracer
    )


def clean(database_file, result_path):

    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


class TestFitInterferometerAgg:

    # def test__fit_interferometer_generator_from_aggregator(
    #     interferometer_7, mask_2d_7x7, samples, model
    # ):
    #
    #     path_prefix = "aggregator_fit_interferometer_gen"
    #
    #     database_file = path.join(conf.instance.output_path, "fit_interferometer.sqlite")
    #     result_path = path.join(conf.instance.output_path, path_prefix)
    #
    #     clean(database_file=database_file, result_path=result_path)
    #
    #     search = al.m.MockSearch(samples=samples)
    #     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    #     analysis = al.AnalysisInterferometer(dataset=interferometer_7)
    #     search.fit(model=model, analysis=analysis)
    #
    #     agg = af.Aggregator.from_database(filename=database_file)
    #     agg.add_directory(directory=result_path)
    #
    #     fit_interferometer_agg = al.agg.FitImagingAgg(aggregator=agg)
    #     fit_interferometer_gen = fit_interferometer_agg.max_log_likelihood()
    #
    #     for fit_interferometer in fit_interferometer_gen:
    #         assert (fit_interferometer.visibilities == interferometer_7.visibilities).all()
    #         assert (fit_interferometer.interferometer.real_space_mask == mask_2d_7x7).all()
    #
    #     clean(database_file=database_file, result_path=result_path)

    def test__fit_interferometer_randomly_drawn_via_pdf_gen_from(
        self, interferometer_7, samples, model
    ):

        path_prefix = "aggregator_fit_interferometer_gen"

        database_file = path.join(
            conf.instance.output_path, "fit_interferometer.sqlite"
        )
        result_path = path.join(conf.instance.output_path, path_prefix)

        clean(database_file=database_file, result_path=result_path)

        search = al.m.MockSearch(
            samples=samples, result=al.m.MockResult(model=model, samples=samples)
        )
        search.paths = af.DirectoryPaths(path_prefix=path_prefix)
        analysis = al.AnalysisInterferometer(dataset=interferometer_7)
        search.fit(model=model, analysis=analysis)

        agg = af.Aggregator.from_database(filename=database_file)
        agg.add_directory(directory=result_path)

        fit_interferometer_agg = al.agg.FitInterferometerAgg(aggregator=agg)
        fit_interferometer_pdf_gen = fit_interferometer_agg.randomly_drawn_via_pdf_gen_from(
            total_samples=2
        )

        i = 0

        for fit_interferometer_gen in fit_interferometer_pdf_gen:

            for fit_interferometer in fit_interferometer_gen:
                i += 1

                assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
                assert fit_interferometer.tracer.galaxies[0].light.centre == (
                    10.0,
                    10.0,
                )
                assert fit_interferometer.tracer.galaxies[1].redshift == 1.0

        assert i == 2

        clean(database_file=database_file, result_path=result_path)

    def test__fit_interferometer_all_above_weight_gen(
        self, interferometer_7, samples, model
    ):

        path_prefix = "aggregator_fit_interferometer_gen"

        database_file = path.join(
            conf.instance.output_path, "fit_interferometer.sqlite"
        )
        result_path = path.join(conf.instance.output_path, path_prefix)

        clean(database_file=database_file, result_path=result_path)

        search = al.m.MockSearch(
            samples=samples, result=al.m.MockResult(model=model, samples=samples)
        )
        search.paths = af.DirectoryPaths(path_prefix=path_prefix)
        analysis = al.AnalysisInterferometer(dataset=interferometer_7)
        search.fit(model=model, analysis=analysis)

        agg = af.Aggregator.from_database(filename=database_file)
        agg.add_directory(directory=result_path)

        fit_interferometer_agg = al.agg.FitInterferometerAgg(aggregator=agg)
        fit_interferometer_pdf_gen = fit_interferometer_agg.all_above_weight_gen_from(
            minimum_weight=-1.0
        )

        i = 0

        for fit_interferometer_gen in fit_interferometer_pdf_gen:

            for fit_interferometer in fit_interferometer_gen:

                i += 1

                if i == 1:
                    assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
                    assert fit_interferometer.tracer.galaxies[0].light.centre == (
                        1.0,
                        1.0,
                    )
                    assert fit_interferometer.tracer.galaxies[1].redshift == 1.0

                if i == 2:
                    assert fit_interferometer.tracer.galaxies[0].redshift == 0.5
                    assert fit_interferometer.tracer.galaxies[0].light.centre == (
                        10.0,
                        10.0,
                    )
                    assert fit_interferometer.tracer.galaxies[1].redshift == 1.0

        assert i == 2

        clean(database_file=database_file, result_path=result_path)
