import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_only'
test_name = "lens_x1_galaxy_hyper"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='LSST', data_name='lens_only_bulge_and_disk')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):
    phase1 = ph.LensPlanePhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    phase1h = ph.LensLightHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase1h".format(test_name))

    class LensHyperPhase(ph.LensPlaneHyperPhase):
        def pass_priors(self, previous_results):
            phase1_results = previous_results[-1]
            phase1h_results = previous_results[-1].hyper
            self.lens_galaxies = phase1_results.variable.lens_galaxies
            self.lens_galaxies[0].hyper_galaxy = phase1h_results.constant.lens_galaxies[0].hyper_galaxy

    phase2 = LensHyperPhase(lens_galaxies=[], optimizer_class=nl.MultiNest,
                            phase_name="{}/phase2".format(test_name))

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase1h, phase2)


if __name__ == "__main__":
    pipeline()
