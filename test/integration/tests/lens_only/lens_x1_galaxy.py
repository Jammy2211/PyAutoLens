import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration import integration_util
from test.simulation import simulation_util

test_type = "lens_only"
test_name = "lens_x1_galaxy"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"


def pipeline(optimizer_class=af.MultiNest):
    af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    ccd_data = simulation_util.load_test_ccd_data(
        data_type="lens_only_dev_vaucouleurs", data_resolution="LSST"
    )

    make_pipeline(
        optimizer_class=optimizer_class
    ).run(
        data=ccd_data
    )


def make_pipeline(
        optimizer_class=af.MultiNest
):
    phase1 = phase_imaging.LensPlanePhase(
        phase_name="phase_1",
        phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
