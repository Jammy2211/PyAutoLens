import os

import autofit as af
from test_autolens.integration import integration_util
from test_autolens.simulate.imaging import simulate_util
from autofit.optimize.non_linear.mock_nlo import MockNLO


def run(
    module,
    test_name=None,
    optimizer_class=af.MultiNest,
    config_folder="config",
    positions=None,
):
    test_name = test_name or module.test_name
    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "output/imaging/"
    config_path = test_path + config_folder
    af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    imaging = simulate_util.load_test_imaging(
        data_type=module.data_type, data_resolution=module.data_resolution
    )

    module.make_pipeline(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        optimizer_class=optimizer_class,
    ).run(dataset=imaging, positions=positions)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        optimizer_class=MockNLO,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        optimizer_class=af.MultiNest,
        config_folder="config_mock",
    )
