import os
import shutil

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source'
test_name = "lens_mass_x1_source_x1_positions"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/' + test_type
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='LSST', data_type='no_lens_light_and_source_smooth')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data, positions=[[[0.8, 0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]]])


def make_pipeline(test_name):

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, use_positions=True,
                                     phase_name="{}/phase1".format(test_name))

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
