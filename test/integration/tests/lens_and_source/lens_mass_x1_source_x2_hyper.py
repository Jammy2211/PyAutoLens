import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source'
test_name = "lens_mass_x1_source_x2_hyper"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def run_pipeline():
    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_light_and_source_smooth', data_resolution='LSST')
    pipeline = make_pipeline()
    pipeline.run(data=ccd_data)


def make_pipeline():
    phase1 = ph.LensSourcePlanePhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                                     lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source_0=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    phase2 = ph.HyperGalaxyPhase(phase_name='phase_2_hyper', phase_folders=[test_type, test_name])

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    run_pipeline()
