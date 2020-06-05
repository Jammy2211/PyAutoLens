import os
from autoconf import conf
import autofit as af
import autolens as al
from test_autolens.simulators.imaging import instrument_util


def run(
    module,
    test_name=None,
    non_linear_class=af.MultiNest,
    config_folder="config",
    mask=None,
):

    test_name = test_name or module.test_name
    test_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = f"{test_path}/output/imaging"
    config_path = f"{test_path}/{config_folder}"
    conf.instance = conf.Config(config_path=config_path, output_path=output_path)

    imaging = instrument_util.load_test_imaging(
        data_label=module.data_label, instrument=module.instrument
    )

    if mask is None:
        mask = al.Mask.circular(
            shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
        )

    module.make_pipeline(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        non_linear_class=non_linear_class,
    ).run(dataset=imaging, mask=mask)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        non_linear_class=af.MockNLO,
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
