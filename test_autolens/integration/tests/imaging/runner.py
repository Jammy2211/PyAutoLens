import os
from os import path
from autoconf import conf
import autofit as af
import autolens as al
from test_autolens.simulators.imaging import instrument_util


def run(module, test_name=None, config_folder="config", mask=None):

    test_name = test_name or module.test_name
    test_path = path.join("{}".format(os.path.dirname(os.path.realpath(__file__))), "..", "..")
    output_path = path.join(test_path, "output", "imaging")
    config_path = path.join(test_path, config_folder)
    conf.instance.push(new_path=config_path, output_path=output_path)

    imaging = instrument_util.load_test_imaging(
        dataset_name=module.dataset_name, instrument=module.instrument
    )

    if mask is None:
        mask = al.Mask2D.circular(
            shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
        )

    info = {"Test": 100}

    module.make_pipeline(
        name=test_name, path_prefix=path.join(module.test_type, test_name)
    ).run(dataset=imaging, mask=mask, info=info)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=path.join(module.test_name, "_mock"),
        search=af.MockSearch,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=path.join(module.test_name, "_nest"),
        search=af.DynestyStatic(),
        config_folder="config_mock",
    )
