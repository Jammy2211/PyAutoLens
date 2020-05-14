import os

import autofit as af
from autofit.optimize.non_linear.mock_nlo import MockNLO
from test_autolens.integration import integration_util
from test_autolens.simulate.interferometer import simulate_util


def run(
    module,
    test_name=None,
    non_linear_class=af.MultiNest,
    config_folder="config",
    positions=None,
):
    test_name = test_name or module.test_name
    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "output/interferometer/"
    config_path = test_path + config_folder
    conf.instance = conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    interferometer = simulate_util.load_test_interferometer(
        data_label=module.data_label, instrument=module.instrument
    )

    module.make_pipeline_no_lens_light(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        non_linear_class=non_linear_class,
    ).run(dataset=interferometer, positions=positions)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        non_linear_class=MockNLO,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        non_linear_class=af.MultiNest,
        config_folder="config_mock",
    )
