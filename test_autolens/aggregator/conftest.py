from os import path
import os
import pytest
import shutil

import autofit as af
import autolens as al
from autofit.non_linear.samples import Sample


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.Sersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.Sersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = al.Galaxy(redshift=0.5, light=al.lp.Sersic(centre=(0.0, 1.0)))
    galaxy_1 = al.Galaxy(redshift=1.0, light=al.lp.Sersic())

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


@pytest.fixture(name="analysis")
def make_analysis(masked_imaging_7x7):
    def null(paths: af.DirectoryPaths, result):
        pass

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.save_results_for_aggregator = null

    return analysis


def clean(database_file, result_path):

    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)
