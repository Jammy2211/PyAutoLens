import os
import shutil

from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'model_mapper'
test_name = "prior_limits"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_dev_vaucouleurs', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, results):
            self.lens_galaxies.lens.sersic.centre_0 = 0.0
            self.lens_galaxies.lens.sersic.centre_1 = 0.0
            self.lens_galaxies.lens.sersic.axis_ratio = prior.UniformPrior(lower_limit=-0.5, upper_limit=0.1)
            self.lens_galaxies.lens.sersic.phi = 90.0
            self.lens_galaxies.lens.sersic.intensity = prior.UniformPrior(lower_limit=-0.5, upper_limit=0.1)
            self.lens_galaxies.lens.sersic.effective_radius = 1.3
            self.lens_galaxies.lens.sersic.sersic_index = 3.0

    phase1 = MMPhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.sersic.intensity = results.from_phase('phase_1').variable.lens.sersic.intensity
            self.lens_galaxies.lens = results.from_phase('phase_1').variable.lens

    phase2 = MMPhase(
        phase_name='phase_2', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
