import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_positions
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util

test_type = 'positions'
test_name = "positions_phase"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    pipeline = make_pipeline(test_name=test_name)

    pipeline.run(positions=[[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]], pixel_scale=0.05)


def make_pipeline(test_name):

    phase1 = phase_positions.PhasePositions(
        phase_name="phase1", phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, mass=mp.SphericalIsothermal)),
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.1

    return pl.PipelinePositions(test_name, phase1)


if __name__ == "__main__":
    pipeline()
