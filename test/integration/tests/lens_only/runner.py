import os

import autofit as af
from test.integration import integration_util
from test.simulation import simulation_util


def run(
        module,
        test_name=None,
        optimizer_class=af.MultiNest,
):
    test_name = test_name or module.test_name
    test_path = "{}/../../".format(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
    output_path = test_path + "output/"
    config_path = test_path + "config"
    af.conf.instance = af.conf.Config(
        config_path=config_path,
        output_path=output_path
    )
    integration_util.reset_paths(
        test_name=test_name,
        output_path=output_path
    )

    ccd_data = simulation_util.load_test_ccd_data(
        data_type=module.data_type,
        data_resolution=module.data_resolution
    )

    module.make_pipeline(
        name=test_name,
        phase_folders=[module.test_type, module.test_name],
        optimizer_class=optimizer_class
    ).run(
        data=ccd_data
    )
