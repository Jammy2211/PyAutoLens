import os
from os import path
import pytest
import shutil

from autoconf import conf
import autofit as af
import autolens as al
from autoconf.conf import with_config
from autofit.non_linear.samples import Sample


@pytest.fixture(autouse=True)
def set_test_mode():
    os.environ["PYAUTOFIT_TEST_MODE"] = "1"
    yield
    del os.environ["PYAUTOFIT_TEST_MODE"]


def clean(database_file):
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    if path.exists(database_sqlite):
        os.remove(database_sqlite)

    result_path = path.join(conf.instance.output_path, database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def aggregator_from(database_file, analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)

    clean(database_file=database_file)

    search = al.m.MockSearch(
        samples=samples, result=al.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=database_file)
    search.fit(model=model, analysis=analysis)

    analysis.modify_before_fit(paths=search.paths, model=model)
    analysis.visualize_before_fit(paths=search.paths, model=model)

    database_file = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    return agg


@pytest.fixture(name="model")
def make_model():
    dataset_model = af.Model(al.DatasetModel)
    dataset_model.background_sky_level = af.UniformPrior(
        lower_limit=0.5, upper_limit=1.5
    )

    return af.Collection(
        dataset_model=dataset_model,
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.Sersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.Sersic),
        ),
    )


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return al.m.MockSamples(
        model=model,
        sample_list=sample_list,
        prior_means=[1.0] * model.prior_count,
    )
